"""Evaluate summoners by existing 4-lane TCI regressors.

Uses each match's FINAL AVAILABLE 5-minute checkpoint once, avoiding
artificially weighting long matches. TOP/JUNGLE/MIDDLE are individual rankings;
BOTTOM represents an ADC+SUPPORT DUO, not individual skill.
Scores are model predictions of relative gold+level advantage, NOT calibrated skill.

Run:
    python evaluate_tci_summoners.py
    python evaluate_tci_summoners.py --split all --recent 20 --min-games 5
    python evaluate_tci_summoners.py --data-dir tci_supervised --model-dir tci_model_results --match-dir match_data
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

LANES = ("top", "jungle", "middle", "bottom")
SPLITS = ("train", "val", "test")


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_match_metadata(folder):
    """Read match_*.json, including batch files containing lists of games."""
    lookup = {}
    files = sorted(folder.glob("match_*.json"))
    if not files:
        raise FileNotFoundError(f"No match_*.json files found in {folder}")
    for file in files:
        with file.open(encoding="utf-8-sig") as f:
            raw = json.load(f)
        matches = raw if isinstance(raw, list) else [raw]
        for match in matches:
            if not isinstance(match, dict) or "metadata" not in match:
                continue
            mid = str(match["metadata"].get("matchId", ""))
            if not mid:
                continue
            info = match.get("info", {})
            members = {}
            for p in info.get("participants", []):
                pid = int(p.get("participantId", -1))
                if pid < 1:
                    continue
                puuid = str(p.get("puuid") or "")
                game_name = str(p.get("riotIdGameName") or p.get("summonerName") or "")
                tag = str(p.get("riotIdTagline") or "")
                label = (game_name + ("#" + tag if tag else "")) if game_name else puuid[:12]
                members[pid] = {"puuid": puuid, "name": label, "position": p.get("teamPosition", "")}
            lookup[mid] = {"members": members,
                           "gameCreation": int(info.get("gameCreation") or 0)}
    if not lookup:
        raise ValueError(f"No match records found in {folder}")
    return lookup


def identifiers(match, participant_ids, lane):
    """A bot identity is the ordered ADC/SPT pair of PUUIDs."""
    try:
        ids = [int(s.strip()) for s in str(participant_ids).split(";") if s.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid participantIds: {participant_ids!r}") from exc
    players = [match["members"].get(pid) for pid in ids]
    if not players or any(p is None or not p["puuid"] for p in players):
        raise ValueError(f"Missing participant metadata for IDs {ids}")
    if lane == "bottom":
        by_position = {p["position"]: p for p in players}
        if len(ids) != 2 or set(by_position) != {"BOTTOM", "UTILITY"}:
            raise ValueError(f"Expected ADC+SPT for bottom, got {[p['position'] for p in players]}")
        adc, spt = by_position["BOTTOM"], by_position["UTILITY"]
        return "DUO:" + adc["puuid"] + "|" + spt["puuid"], adc["name"] + " + " + spt["name"], "BOT_DUO"
    if len(ids) != 1:
        raise ValueError(f"Expected one player on {lane}: {ids}")
    return players[0]["puuid"], players[0]["name"], "INDIVIDUAL"


def predict_lane(lane, source, models, split):
    splits = SPLITS if split == "all" else (split,)
    chunks = []
    for part in splits:
        file = source / f"{lane}_{part}.csv"
        if not file.exists():
            raise FileNotFoundError(f"Missing {file}. Run prepare_tci_targets_gold_level_v2.py first.")
        df = pd.read_csv(file, dtype={"matchId": str, "participantIds": str})
        df["data_split"] = part
        chunks.append(df)
    data = pd.concat(chunks, ignore_index=True)
    if data.empty:
        raise ValueError(f"Empty dataset for {lane}")
    if data.duplicated(["matchId", "minute", "side"]).any():
        raise ValueError(f"Duplicated observations for {lane}")
    model_file = models / f"{lane}_best_model.joblib"
    if not model_file.exists():
        raise FileNotFoundError(f"Missing {model_file}. Run train_tci_models.py first.")
    # Never load joblib/pickle files from an untrusted source.
    bundle = joblib.load(model_file)
    features = bundle["features"]
    if any(word in f.lower() for f in features for word in ("gold", "level", "xp")):
        raise ValueError(f"Target leakage found in {lane} model features")
    if any(f not in data for f in features):
        raise ValueError(f"Missing features for {lane}: {set(features) - set(data)}")
    numeric = data[features].apply(pd.to_numeric, errors="coerce")
    data["predicted_tci"] = bundle["pipeline"].predict(numeric)
    if not np.isfinite(data["predicted_tci"]).all():
        raise ValueError("Non-finite model predictions")
    data["lane"] = lane.upper()
    data["model_name"] = bundle["model_name"]
    # Once per lane, per side, per match: use last available checkpoint.
    data = (data.sort_values(["matchId", "side", "minute"])
                .drop_duplicates(["matchId", "side"], keep="last").copy())
    return data[["matchId", "lane", "side", "participantIds", "minute",
                 "data_split", "win", "model_name", "predicted_tci"]]


def summarize_games(games, recent, min_games):
    result = []
    for (lane, key), group in games.groupby(["lane", "competitor_id"], sort=False):
        last = group.sort_values(["gameCreation", "matchId"], ascending=[False, False]).head(recent)
        result.append({
            "lane": lane, "competitor_id": key,
            "competitor_name": last["competitor_name"].iloc[0],
            "unit": last["unit"].iloc[0], "games": len(last),
            "mean_tci": round(float(last["predicted_tci"].mean()), 5),
            "median_tci": round(float(last["predicted_tci"].median()), 5),
            "win_rate": round(float(last["win"].mean()), 4),
            "first_game_ms": int(last["gameCreation"].min()),
            "last_game_ms": int(last["gameCreation"].max()),
            "eligible": len(last) >= min_games,
        })
    leaders = pd.DataFrame(result)
    if leaders.empty:
        raise ValueError("No summoners resolved")
    leaders["lane_percentile"] = np.nan
    leaders["grade"] = "UNRANKED"
    for lane in leaders["lane"].unique():
        eligible = (leaders["lane"] == lane) & leaders["eligible"]
        n = int(eligible.sum())
        if n < 5:
            print(f"[WARN] {lane}: only {n} eligible competitors; suppressing grades/percentiles")
            continue
        # Higher average is better; percentile is position within same lane only.
        ranks = leaders.loc[eligible, "mean_tci"].rank(pct=True, method="average") * 100
        leaders.loc[eligible, "lane_percentile"] = ranks
        leaders.loc[eligible, "grade"] = ranks.map(
            lambda v: "S" if v >= 90 else "A" if v >= 70 else "B" if v >= 40
            else "C" if v >= 15 else "D")
    return leaders.sort_values(["lane", "eligible", "mean_tci"], ascending=[True, False, False])


def main():
    parser = argparse.ArgumentParser(description="Summoner and bot-duo TCI ranking (provisional)")
    parser.add_argument("--data-dir", default="tci_supervised")
    parser.add_argument("--model-dir", default="tci_model_results")
    parser.add_argument("--match-dir", default="match_data")
    parser.add_argument("--output-dir", default="tci_summoner_ranking")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--recent", type=int, default=20, help="Max recent matches per competitor")
    parser.add_argument("--min-games", type=int, default=5, help="Minimum matches for grades")
    args = parser.parse_args()
    if args.recent < 1 or args.min_games < 1:
        parser.error("--recent and --min-games must be positive")
    root = Path(__file__).resolve().parent
    source = resolve(root, args.data_dir)
    modeldir = resolve(root, args.model_dir)
    matchdir = resolve(root, args.match_dir)
    output = resolve(root, args.output_dir)
    metadata = read_match_metadata(matchdir)
    frames = [predict_lane(lane, source, modeldir, args.split) for lane in LANES]
    all_games = pd.concat(frames, ignore_index=True)
    known = all_games["matchId"].isin(metadata)
    if not known.all():
        missing = sorted(all_games.loc[~known, "matchId"].unique())
        raise ValueError(f"Missing original match JSONs for {len(missing)} match IDs; first 5: {missing[:5]}")
    records = []
    for row in all_games.itertuples(index=False):
        match = metadata[row.matchId]
        key, name, unit = identifiers(match, row.participantIds, row.lane.lower())
        records.append({"matchId": row.matchId, "lane": row.lane, "side": row.side,
                        "competitor_id": key, "competitor_name": name, "unit": unit,
                        "minute": int(row.minute), "win": int(row.win),
                        "predicted_tci": float(row.predicted_tci),
                        "gameCreation": match["gameCreation"], "data_split": row.data_split,
                        "model_name": row.model_name})
    games = pd.DataFrame(records)
    leaders = summarize_games(games, args.recent, args.min_games)
    output.mkdir(parents=True, exist_ok=True)
    games.sort_values(["lane", "matchId", "side"]).to_csv(
        output / "tci_game_scores.csv", index=False, encoding="utf-8-sig")
    leaders.to_csv(output / "tci_summoner_leaderboard.csv", index=False, encoding="utf-8-sig")
    for lane in LANES:
        sub = leaders[leaders["lane"] == lane.upper()]
        sub.to_csv(output / f"{lane}_ranking.csv", index=False, encoding="utf-8-sig")
        eligible = sub[sub["eligible"]]
        print(f"{lane.upper():7s}: {len(sub)} competitors, {len(eligible)} with >= {args.min_games} matches")
        if not eligible.empty:
            print(eligible[["competitor_name", "games", "mean_tci", "grade"]].head(5).to_string(index=False))
    print(f"\nSaved CSVs to: {output}")
    print("CAUTION: BOT is a duo score, NOT separate ADC/SPT ability. Rankings depend on opponent strength,")
    print("champions, and sample size. They are provisional, not calibrated skill or rank predictions.")
    if args.split == "all":
        print("CAUTION: all includes training matches; for held-out-only diagnostic use --split test.")


if __name__ == "__main__":
    main()

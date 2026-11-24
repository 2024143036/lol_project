"""Riot Match-v5: cumulative four-lane features at 5-minute checkpoints.

Usage:
    python data_process_4lane_cumulative_v2.py --data-dir match_data --output-dir datasets_5min_fixed

Pairs match_N.json with timeline_N.json. Each JSON can contain one match dict
or a list of match dicts; matches are joined by metadata.matchId.

At minute 5, 10, 15... each row contains ONLY values accumulated from
match start to that checkpoint. No 5-minute interval deltas are created.
Never uses final-match stats to create mid-game feature values.
"""
import argparse
import csv
import json
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

LANES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM")
MINUTE_MS = 60_000
WINDOW_MS = 5 * MINUTE_MS
POSITIONS = {"TOP": "TOP", "JUNGLE": "JUNGLE", "MIDDLE": "MIDDLE", "BOTTOM": "BOTTOM", "UTILITY": "BOTTOM"}
SNAP_KEYS = ("cs", "xp", "level", "gold", "damage_to_champions", "total_damage_taken")
EVENT_KEYS = ("kills", "deaths", "assists", "turret_kills", "turret_participation", "dragon_participation", "herald_participation")


def load_objects(path):
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported JSON format: {path}")


def snapshot(pf, pid):
    p = pf.get(str(pid))
    if not p:
        return None
    dmg = p.get("damageStats") or {}
    return {
        "cs": p.get("minionsKilled", 0) + p.get("jungleMinionsKilled", 0),
        "xp": p.get("xp", 0),
        "level": p.get("level", 0),
        "gold": p.get("totalGold", 0),
        "damage_to_champions": dmg.get("totalDamageDoneToChampions", 0),
        "total_damage_taken": dmg.get("totalDamageTaken", 0),
    }


def group_snapshot(pf, pids):
    rows = [snapshot(pf, p) for p in pids]
    if not rows or any(r is None for r in rows):
        return None
    out = {k: sum(r[k] for r in rows) for k in SNAP_KEYS}
    if len(rows) > 1:
        out["level"] /= len(rows)  # bot duo average level
    return out


def event_contributions(ev):
    """Return {pid: set of distinct event contributions}.

    Assistance must be explicitly included in the API event.  Never infer
    participants from team ownership or proximity.
    """
    typ = ev.get("type")
    killer = ev.get("killerId")
    victim = ev.get("victimId")
    helpers = ev.get("assistingParticipantIds") or []
    out = defaultdict(set)
    if typ == "CHAMPION_KILL":
        if isinstance(killer, int) and killer > 0:
            out[killer].add("kills")
        if isinstance(victim, int) and victim > 0:
            out[victim].add("deaths")
        for a in helpers:
            if isinstance(a, int) and a > 0:
                out[a].add("assists")
    elif typ == "BUILDING_KILL" and ev.get("buildingType") == "TOWER_BUILDING":
        if isinstance(killer, int) and killer > 0:
            out[killer].update(("turret_kills", "turret_participation"))
        for a in helpers:
            if isinstance(a, int) and a > 0:
                out[a].add("turret_participation")
    elif typ == "ELITE_MONSTER_KILL":
        monster = ev.get("monsterType")
        if monster == "DRAGON":
            key = "dragon_participation"
        elif monster == "RIFTHERALD":
            key = "herald_participation"
        else:
            return out
        for p in [killer, *helpers]:
            if isinstance(p, int) and p > 0:
                out[p].add(key)
    return out


def produce_match(match, timeline):
    mid = match.get("metadata", {}).get("matchId")
    if mid != timeline.get("metadata", {}).get("matchId"):
        raise ValueError(f"match/timeline ID mismatch: {mid}")
    participants = match["info"]["participants"]
    sides = defaultdict(lambda: defaultdict(list))
    for p in participants:
        pos = p.get("teamPosition")
        if pos in POSITIONS:
            side = "blue" if p["teamId"] == 100 else "red"
            sides[POSITIONS[pos]][side].append(p["participantId"])
    for lane in LANES:
        n = 2 if lane == "BOTTOM" else 1
        if any(len(sides[lane][side]) != n for side in ("blue", "red")):
            return []  # missing / abnormal role assignment: do not silently mislabel

    frames = sorted(timeline["info"]["frames"], key=lambda x: x.get("timestamp", 0))
    duration_ms = int(match["info"]["gameDuration"] * 1000)
    ends = list(range(WINDOW_MS, duration_ms + 1, WINDOW_MS))
    if not ends:
        return []

    # Store every event once, irrespective of frame storage boundaries.
    events = sorted((e for frame in frames for e in frame.get("events", [])
                     if 0 <= e.get("timestamp", -1) <= duration_ms),
                    key=lambda e: e["timestamp"])
    counts = defaultdict(lambda: defaultdict(int))
    rows = []
    ei = 0
    frame_times = [f.get("timestamp", 0) for f in frames]
    for end in ends:
        # Riot timeline snapshots may be slightly AFTER exact minute boundaries.
        # Never label a 4-minute snapshot as the 5-minute snapshot.
        fi = bisect_left(frame_times, end)
        if fi >= len(frames):
            print(f"[WARN] {mid}: missing {end // MINUTE_MS}min snapshot")
            continue
        frame_ts = frame_times[fi]
        if frame_ts - end > 15_000:
            print(f"[WARN] {mid}: {end // MINUTE_MS}min snapshot too late ({frame_ts / MINUTE_MS:.3f})")
            continue
        # Event counts must match the actual numerical snapshot timestamp.
        while ei < len(events) and events[ei]["timestamp"] <= frame_ts:
            ev = events[ei]
            for pid, contributions in event_contributions(ev).items():
                for key in contributions:
                    counts[pid][key] += 1
            ei += 1
        pf = frames[fi].get("participantFrames", {})
        for lane in LANES:
            # Make one snapshot per side, then compute same-time opponent differences.
            snapshots = {}
            for side in ("blue", "red"):
                pids = sides[lane][side]
                snap = group_snapshot(pf, pids)
                if snap is None:
                    break
                event_counts = {
                    k: sum(counts[pid][k] for pid in pids)
                    for k in EVENT_KEYS
                }
                snapshots[side] = {**snap, **event_counts}
            if len(snapshots) != 2:
                continue
            for side, opponent_side in (("blue", "red"), ("red", "blue")):
                pids = sides[lane][side]
                own = snapshots[side]
                opp = snapshots[opponent_side]
                team_win = next(
                    p["win"] for p in participants if p["participantId"] == pids[0]
                )
                row = {
                    "matchId": mid,
                    "lane": lane,
                    "side": side,
                    "participantIds": ";".join(map(str, pids)),
                    "minute": end // MINUTE_MS,
                    "snapshot_minute": round(frame_ts / MINUTE_MS, 3),
                    "win": int(bool(team_win)),
                }
                for k in SNAP_KEYS + EVENT_KEYS:
                    row[f"{k}_cum"] = own[k]
                    row[f"{k}_opponent_cum"] = opp[k]
                    row[f"{k}_diff_cum"] = own[k] - opp[k]
                # Exact turret damage at 5/10/15... is not present in the
                # timeline damageStats. Do not copy game-end totals here.
                row["turret_damage_cum"] = ""
                row["turret_damage_opponent_cum"] = ""
                row["turret_damage_diff_cum"] = ""
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="match_data")
    parser.add_argument("--output-dir", default="datasets_5min_fixed")
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    src, dest = Path(args.data_dir), Path(args.output_dir)
    if not src.is_absolute():
        src = script_dir / src
    if not dest.is_absolute():
        dest = script_dir / dest
    print("TCI cumulative snapshot extractor [fixed 5-minute timestamps, v2]")
    print(f"Data folder: {src}")
    dest.mkdir(parents=True, exist_ok=True)
    all_rows = {lane: [] for lane in LANES}
    matches = 0
    skipped = 0
    for mp in sorted(src.glob("match_*.json")):
        suffix = mp.name[len("match_"):]
        tp = src / f"timeline_{suffix}"
        if not tp.exists():
            print(f"[SKIP] Missing timeline: {tp}")
            continue
        try:
            mm = load_objects(mp)
            tt = {x["metadata"]["matchId"]: x for x in load_objects(tp)}
            for match in mm:
                mid = match["metadata"]["matchId"]
                if mid not in tt:
                    skipped += 1
                    continue
                rows = produce_match(match, tt[mid])
                if not rows:
                    skipped += 1
                    continue
                matches += 1
                for r in rows:
                    all_rows[r["lane"]].append(r)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            print(f"[SKIP] {mp.name}: {exc}")
            skipped += 1
    for lane, data in all_rows.items():
        path = dest / f"{lane.lower()}_cumulative_5min.csv"
        if data:
            with path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                writer.writeheader()
                writer.writerows(data)
        elif path.exists():
            path.unlink()  # Prevent stale output from being mistaken for a fresh run.
        print(f"{lane}: {len(data)} rows -> {path if data else 'no valid records'}")
    print(f"Processed matches: {matches}; skipped matches: {skipped}")
    print("Verify CSV: minute=5 should have snapshot_minute near 5.0 (not 4.0).")


if __name__ == "__main__":
    main()

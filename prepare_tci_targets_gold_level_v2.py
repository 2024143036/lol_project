"""Build supervised TCI datasets from four 5-minute cumulative CSVs.

Target y = 0.5*(gold_diff_cum / sigma_gold) + 0.5*(level_diff_cum / sigma_level).
Sigma is estimated ONLY from training matches for each lane.
Division without subtracting a mean keeps blue/red target scores anti-symmetric.

Input: datasets_5min_fixed/{top,jungle,middle,bottom}_cumulative_5min.csv
Output: tci_supervised/{lane}_{train,val,test}.csv and target_scaling.json
No model is trained by this script.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

LANES = ('top', 'jungle', 'middle', 'bottom')
BASE_FEATURES = [
    'cs', 'kills', 'deaths', 'assists',
    'turret_kills', 'turret_participation',
    'damage_to_champions', 'total_damage_taken',
]
EXTRA_JUNGLE = ['dragon_participation', 'herald_participation']
META_COLS = ['matchId', 'lane', 'side', 'participantIds', 'minute', 'snapshot_minute', 'win']
TARGET_COLS = ['gold_diff_cum', 'level_diff_cum']


def split_matches(match_ids, seed):
    ids = np.array(sorted(set(match_ids)), dtype=str)
    if len(ids) < 7:
        raise ValueError(f'At least 7 independent matches needed for 70/15/15 split, got {len(ids)}')
    # Exactly the same match IDs are assigned to the same partition in all lanes.
    rng = np.random.default_rng(seed)
    ids = rng.permutation(ids)
    n = len(ids)
    n_test = max(1, int(round(n * 0.15)))
    n_val = max(1, int(round(n * 0.15)))
    if n - n_val - n_test < 2:
        raise ValueError('Too few train matches')
    return {
        'train': set(ids[:n-n_val-n_test]),
        'val': set(ids[n-n_val-n_test:n-n_test]),
        'test': set(ids[n-n_test:]),
    }


def main():
    parser = argparse.ArgumentParser(description='Make 4-lane supervised TCI training targets')
    parser.add_argument('--input-dir', default='datasets_5min_fixed')
    parser.add_argument('--output-dir', default='tci_supervised')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    if not in_dir.is_absolute():
        in_dir = root / in_dir
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = {}
    for lane in LANES:
        path = in_dir / f'{lane}_cumulative_5min.csv'
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}. Generate four lane CSVs first.')
        df = pd.read_csv(path, encoding='utf-8-sig', dtype={'matchId': str})
        if df.empty:
            raise ValueError(f'Empty dataset: {path}')
        names = BASE_FEATURES + (EXTRA_JUNGLE if lane == 'jungle' else [])
        # Use same-time opponent differences, always measured from own side.
        cols = [f'{n}_diff_cum' for n in names]
        required = META_COLS + TARGET_COLS + cols
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f'{lane}: missing columns: {missing}')
        if df[required].isna().any().any():
            raise ValueError(f'{lane}: found nulls in required columns')
        if (df['minute'] % 5 != 0).any() or (df['minute'] < 5).any():
            raise ValueError(f'{lane}: not all rows are 5-minute checkpoints')
        if (df['snapshot_minute'] - df['minute']).abs().gt(0.25).any():
            raise ValueError(f'{lane}: old 4/9/14-minute snapshots detected; rerun v2 extractor')
        if df.duplicated(['matchId', 'minute', 'side']).any():
            raise ValueError(f'{lane}: duplicate match/minute/side rows')
        for (mid,minute), group in df.groupby(['matchId','minute']):
            if set(group['side']) != {'blue','red'} or len(group) != 2:
                raise ValueError(f'{lane}: match {mid} minute {minute} missing a side')
            for target in TARGET_COLS:
                if not np.isclose(group[target].sum(), 0):
                    raise ValueError(f'{lane}: {target} sides not symmetric in {mid}/{minute}')
        # Keep only designed feature columns; no gold/level (raw or diff) in X.
        cols = [c for c in cols if c not in TARGET_COLS]
        frames[lane] = df[META_COLS + cols + TARGET_COLS].copy()

    # Restrict all lanes to matches represented in ALL four datasets.
    shared_ids = set.intersection(*(set(df['matchId']) for df in frames.values()))
    if not shared_ids:
        raise ValueError('No common matchId across all four lanes')
    parts = split_matches(shared_ids, args.seed)
    scales = {'target_definition': '0.5*(gold_diff_cum/sd_gold)+0.5*(level_diff_cum/sd_level)',
              'note': 'Std fitted on training matches only. Gold and level diff are excluded from input X.',
              'seed': args.seed,
              'splits': {k: len(v) for k,v in parts.items()}, 'lanes': {}}
    print(f'Shared matches: {len(shared_ids)}. Train/val/test matches: ' +
          '/'.join(str(len(parts[x])) for x in ('train','val','test')))
    for lane, df in frames.items():
        df = df[df.matchId.isin(shared_ids)].copy()
        train_rows = df[df.matchId.isin(parts['train'])]
        # Standard deviation of all own-side / opponent-side match records.
        # Paired opposing rows are symmetric around zero by construction.
        sd_gold = float(train_rows['gold_diff_cum'].std(ddof=0))
        sd_level = float(train_rows['level_diff_cum'].std(ddof=0))
        if sd_gold <= 0 or sd_level <= 0:
            raise ValueError(f'{lane}: zero variance in target component')
        df['y_tci'] = 0.5 * (df.gold_diff_cum / sd_gold + df.level_diff_cum / sd_level)
        # Gold and level are ONLY used to calculate y. Drop from final input files.
        df = df.drop(columns=TARGET_COLS)
        features = [c for c in df if c.endswith('_diff_cum')]
        forbidden = {'gold_diff_cum', 'level_diff_cum', 'xp_diff_cum'}
        if forbidden.intersection(features):
            raise ValueError(f'{lane}: target/leaked feature remains: {forbidden.intersection(features)}')
        assert 'gold_diff_cum' not in features and 'level_diff_cum' not in features
        scales['lanes'][lane] = {'sd_gold': sd_gold, 'sd_level': sd_level,
                                 'features': features, 'rows': int(len(df))}
        for split, ids in parts.items():
            output = out_dir / f'{lane}_{split}.csv'
            chunk = df[df.matchId.isin(ids)]
            chunk.to_csv(output, index=False, encoding='utf-8-sig')
            print(f'  {lane:7} {split:5}: {len(chunk):4} rows -> {output.name}')
    (out_dir / 'target_scaling.json').write_text(json.dumps(scales, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nDone: 4-lane X + y data, split by matchId. No model trained yet.')

if __name__ == '__main__':
    main()

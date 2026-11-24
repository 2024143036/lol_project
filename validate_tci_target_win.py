"""Compare win-prediction evidence for 3 proposed TCI target definitions.

Run from the lol_project directory:
    python validate_tci_target_win.py --input-dir datasets_5min_fixed

Only cumulative feature CSVs from data_process_4lane_cumulative_v2.py are used.
The supervised TCI CSVs cannot be used: they deliberately omit the gold and
level gaps needed here. Blue-side records only (one label per match/minute)
prevent treating mirrored red/blue observations as independent matches.

Outputs validation and test metrics.
Target selection is based on validation performance.
Test data is used only for final evaluation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LANES = ('top', 'jungle', 'middle', 'bottom')
CANDIDATES = ('gold_gap', 'level_gap', 'gold_plus_level')


def split_ids(ids, seed):
    ids = np.array(sorted(set(ids)), dtype=str)
    if len(ids) < 7:
        raise ValueError(f'At least 7 shared matches required; got {len(ids)}')
    ids = np.random.default_rng(seed).permutation(ids)
    n = len(ids)
    n_test, n_val = max(1, round(n * .15)), max(1, round(n * .15))
    return {'train': set(ids[:n - n_test - n_val]),
            'val': set(ids[n - n_test - n_val:n - n_test]),
            'test': set(ids[n - n_test:])}


def score_frame(df, candidate, s_gold, s_level):
    gold = df['gold_diff_cum'].to_numpy(float) / s_gold
    level = df['level_diff_cum'].to_numpy(float) / s_level
    if candidate == 'gold_gap':
        value = gold
    elif candidate == 'level_gap':
        value = level
    else:
        value = .5 * gold + .5 * level
    return value.reshape(-1, 1)


def evaluate(model, x, y):
    probability = model.predict_proba(x)[:, 1]
    predicted = (probability >= .5).astype(int)
    return {
        'accuracy': round(float(accuracy_score(y, predicted)), 4),
        'roc_auc': round(float(roc_auc_score(y, probability)), 4) if len(np.unique(y)) == 2 else np.nan,
        'log_loss': round(float(log_loss(y, probability, labels=[0, 1])), 4),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate TCI target candidates on win prediction')
    parser.add_argument('--input-dir', default='datasets_5min_fixed')
    parser.add_argument('--output-dir', default='target_win_validation')
    parser.add_argument('--lanes', nargs='+', choices=LANES, default=list(LANES))
    parser.add_argument('--minutes', nargs='+', type=int, default=[5, 10, 15, 20])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    source = Path(args.input_dir)
    output = Path(args.output_dir)
    if not source.is_absolute(): source = root / source
    if not output.is_absolute(): output = root / output
    output.mkdir(parents=True, exist_ok=True)

    raw = {}
    for lane in args.lanes:
        path = source / f'{lane}_cumulative_5min.csv'
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}; use 5-minute cumulative extractor output, NOT tci_supervised')
        df = pd.read_csv(path, dtype={'matchId': str})
        need = ['matchId', 'side', 'minute', 'snapshot_minute', 'win', 'gold_diff_cum', 'level_diff_cum']
        missing = [c for c in need if c not in df]
        if missing:
            raise ValueError(f'{lane}: missing columns {missing}')
        if df[need].isna().any().any():
            raise ValueError(f'{lane}: missing required values')
        if (df['snapshot_minute'] - df['minute']).abs().gt(.25).any():
            raise ValueError(f'{lane}: bad snapshot times (4m for 5m etc). Re-run v2 extractor')
        if df.duplicated(['matchId', 'minute', 'side']).any():
            raise ValueError(f'{lane}: duplicated match/minute/side')
        # Opposing rows share the same game outcome; keep exactly one record per match/time.
        raw[lane] = df.loc[df.side == 'blue'].copy()
    shared = set.intersection(*(set(df.matchId) for df in raw.values()))
    splits = split_ids(shared, args.seed)
    print('Common matches:', len(shared), '| split:', {k: len(v) for k, v in splits.items()})

    all_rows = []
    scale_records = {}
    for lane, frame in raw.items():
        frame = frame[frame.matchId.isin(shared)]
        for minute in args.minutes:
            data = frame[frame.minute == minute].copy()
            partitions = {name: data[data.matchId.isin(ids)] for name, ids in splits.items()}
            tr, va, te = (partitions[k] for k in ('train', 'val', 'test'))
            if any(len(x) == 0 for x in (tr, va, te)):
                print(f'SKIP {lane} {minute}m: train/val/test missing match records')
                continue
            if len(tr.win.unique()) < 2:
                print(f'SKIP {lane} {minute}m: training set has only one win class')
                continue
            sd_gold = float(tr.gold_diff_cum.std(ddof=0))
            sd_level = float(tr.level_diff_cum.std(ddof=0))
            # Only the training partition is allowed to determine scaling.
            if sd_gold <= 0 or sd_level <= 0:
                print(f'SKIP {lane} {minute}m: zero standard deviation')
                continue
            scale_records[f'{lane}_{minute}m'] = {'gold_sd': sd_gold, 'level_sd': sd_level}
            for candidate in CANDIDATES:
                xtr = score_frame(tr, candidate, sd_gold, sd_level)
                model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=args.seed))
                model.fit(xtr, tr.win.astype(int))
                for split in ('val', 'test'):
                    part = partitions[split]
                    scores = evaluate(model, score_frame(part, candidate, sd_gold, sd_level), part.win.astype(int))
                    row = {'lane': lane, 'minute': minute, 'candidate': candidate,
                           'split': split, 'train_matches': len(tr),
                           'evaluated_matches': len(part), **scores}
                    all_rows.append(row)
                    print(f'{lane:7} {minute:2}m {candidate:16} {split:4}: '
                          f'accuracy={scores["accuracy"]:.3f} AUC={scores["roc_auc"]} n={len(part)}')
    if not all_rows:
        raise ValueError('No evaluations completed. Need more games or different checkpoints')
    result = pd.DataFrame(all_rows)
    result.to_csv(output / 'win_prediction_comparison.csv', index=False, encoding='utf-8-sig')
    # Target selection must be determined using validation only.
    validation = result[result.split == 'val'].copy()
    validation['auc_sort'] = validation.roc_auc.fillna(-1)
    winners = (validation.sort_values(['lane', 'minute', 'auc_sort', 'accuracy', 'log_loss'],
                                      ascending=[True, True, False, False, True])
               .drop_duplicates(['lane', 'minute'])
               .drop(columns=['auc_sort']))
    winners.to_csv(output / 'validation_winners.csv', index=False, encoding='utf-8-sig')
    (output / 'experiment_info.json').write_text(json.dumps({
        'shared_matches': len(shared), 'splits': {k: len(v) for k,v in splits.items()},
        'seed': args.seed, 'minutes': args.minutes, 'lanes': args.lanes,
        'target_candidates': list(CANDIDATES),
        'note': 'One blue-side observation per match and minute. Target ranking is determined using validation ROC-AUC.',
        'scales': scale_records,
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nDONE:', output)
    print('Read validation_winners.csv to see candidates ranked on validation (not test).')

if __name__ == '__main__':
    main()

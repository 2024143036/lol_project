"""Train and evaluate lane-specific TCI regressors, with match-disjoint splits.

Run after prepare_tci_targets_gold_level_v2.py:
    python train_tci_models.py

Reads tci_supervised/{top,jungle,middle,bottom}_{train,val,test}.csv.
Picks each lane's model on VALIDATION RMSE, and evaluates TEST only once.
No model uses gold_diff_cum or level_diff_cum as inputs.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LANES = ('top', 'jungle', 'middle', 'bottom')
FORBIDDEN = {'gold_diff_cum', 'level_diff_cum', 'xp_diff_cum', 'y_tci', 'win',
             'matchId', 'side', 'lane', 'minute', 'snapshot_minute', 'participantIds'}


def metrics(y, pred):
    return {
        'mae': round(float(mean_absolute_error(y, pred)), 6),
        'rmse': round(float(np.sqrt(mean_squared_error(y, pred))), 6),
        'r2': round(float(r2_score(y, pred)), 6) if len(y) > 1 else None,
    }


def make_models(seed, add_xgb):
    models = {
        'LinearRegression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('model', LinearRegression()),
        ]),
        'Ridge': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('model', Ridge(alpha=10.0)),
        ]),
        'RandomForest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestRegressor(n_estimators=150, max_depth=8,
                                            min_samples_leaf=3, random_state=seed,
                                            n_jobs=-1)),
        ]),
    }
    if add_xgb:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError('Install xgboost: python -m pip install xgboost') from exc
        models['XGBoost'] = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', XGBRegressor(n_estimators=250, max_depth=3,
                                   learning_rate=0.05, subsample=0.85,
                                   colsample_bytree=0.9, reg_lambda=5,
                                   objective='reg:squarederror', random_state=seed,
                                   n_jobs=2)),
        ])
    return models


def check_partitions(data, lane):
    ids = [set(df['matchId'].astype(str)) for df in data.values()]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError(f'{lane}: matchId leakage across train/val/test')
    for part, df in data.items():
        if len(df) == 0:
            raise ValueError(f'{lane}: empty {part} data')
        if df['y_tci'].isna().any():
            raise ValueError(f'{lane}: missing target in {part}')
        if df.duplicated(['matchId', 'minute', 'side']).any():
            raise ValueError(f'{lane}: duplicated samples in {part}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='tci_supervised')
    parser.add_argument('--output-dir', default='tci_model_results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-xgboost', action='store_true',
                        help='Use only sklearn models if xgboost is not installed')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    source = Path(args.input_dir)
    dest = Path(args.output_dir)
    if not source.is_absolute(): source = root / source
    if not dest.is_absolute(): dest = root / dest
    dest.mkdir(parents=True, exist_ok=True)

    all_results = []
    best_summary = {}
    for lane in LANES:
        data = {}
        for part in ('train', 'val', 'test'):
            path = source / f'{lane}_{part}.csv'
            if not path.exists():
                raise FileNotFoundError(f'{path} not found: run target preparation first')
            data[part] = pd.read_csv(path, dtype={'matchId': str})
        check_partitions(data, lane)
        features = [c for c in data['train'].columns if c.endswith('_diff_cum')]
        features = [c for c in features if c not in FORBIDDEN]
        if not features: raise ValueError(f'{lane}: no usable features')
        if any(any(x in f.lower() for x in ('gold', 'level', 'xp')) for f in features):
            raise ValueError(f'{lane}: target-related feature detected: {features}')
        if any(not set(features).issubset(df.columns) for df in data.values()):
            raise ValueError(f'{lane}: mismatched feature columns')
        X = {part: df[features].apply(pd.to_numeric, errors='coerce')
             for part, df in data.items()}
        y = {part: df['y_tci'].to_numpy(dtype=float) for part, df in data.items()}
        if X['train'].isna().all().any():
            raise ValueError(f'{lane}: entirely missing training feature')

        candidates = make_models(args.seed, not args.no_xgboost)
        scored = []
        fitted = {}
        for name, model in candidates.items():
            model.fit(X['train'], y['train'])
            pred_train = model.predict(X['train'])
            pred_val = model.predict(X['val'])
            tr = metrics(y['train'], pred_train)
            va = metrics(y['val'], pred_val)
            all_results.append({'lane':lane,'model':name,'split':'train',**tr})
            all_results.append({'lane':lane,'model':name,'split':'val',**va})
            fitted[name] = model
            scored.append((va['rmse'], name))
            print(f'{lane:7s} {name:17s} train RMSE={tr["rmse"]:.4f} | val RMSE={va["rmse"]:.4f}')

        # Select by validation only; keep test data untouched until selected.
        _, winning_name = min(scored)
        best = fitted[winning_name]
        preds = best.predict(X['test'])
        test_metrics = metrics(y['test'], preds)
        all_results.append({'lane':lane,'model':winning_name,'split':'test',**test_metrics})
        predictions = data['test'][['matchId','side','minute','y_tci']].copy()
        predictions['predicted_tci'] = preds
        predictions['error'] = predictions['predicted_tci'] - predictions['y_tci']
        predictions.to_csv(dest / f'{lane}_test_predictions.csv', index=False, encoding='utf-8-sig')
        bundle = {'pipeline':best,'features':features,'lane':lane,'model_name':winning_name,
                  'target':'0.5*(gold_gap/sd_gold)+0.5*(level_gap/sd_level)'}
        joblib.dump(bundle, dest / f'{lane}_best_model.joblib')
        best_summary[lane] = {'model':winning_name, 'test_metrics':test_metrics,
                              'features':features,
                              'matches':{part:int(data[part]['matchId'].nunique())
                                         for part in ('train','val','test')}}
        print(f'   BEST {lane}: {winning_name}; test RMSE={test_metrics["rmse"]:.4f}, '
              f'MAE={test_metrics["mae"]:.4f}, R2={test_metrics["r2"]:.4f}')

    pd.DataFrame(all_results).to_csv(dest / 'model_comparison.csv', index=False, encoding='utf-8-sig')
    (dest / 'best_models_summary.json').write_text(
        json.dumps(best_summary, indent=2, ensure_ascii=False),encoding='utf-8')
    print(f'\nFinished. Saved files in: {dest}')

if __name__ == '__main__':
    main()

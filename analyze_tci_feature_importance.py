"""Analyze feature influence for trained TOP/JUNGLE/MIDDLE/BOTTOM TCI models.

Uses permutation importance on VALIDATION data, never test data.
A feature's importance = increase in RMSE after shuffling that feature.
Positive: useful to the model; zero/negative: little improvement on this set.

Run from the project directory:
  python analyze_tci_feature_importance.py
  python analyze_tci_feature_importance.py --data-dir tci_supervised --model-dir tci_model_results
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

LANES = ('top', 'jungle', 'middle', 'bottom')
LABELS = {
    'cs': 'CS', 'kills': 'Kills', 'deaths': 'Deaths', 'assists': 'Assists',
    'turret_kills': 'Turrets destroyed',
    'turret_participation': 'Turret participation',
    'damage_to_champions': 'Champion damage dealt',
    'total_damage_taken': 'Total damage taken',
    'dragon_participation': 'Dragon participation',
    'herald_participation': 'Herald participation',
}


def display_name(feature):
    return LABELS.get(feature.removesuffix('_diff_cum'), feature)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='tci_supervised')
    parser.add_argument('--model-dir', default='tci_model_results')
    parser.add_argument('--output-dir', default='tci_feature_importance')
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('--repeats must be >= 2')

    root = Path(__file__).resolve().parent
    def location(name):
        p = Path(name)
        return p if p.is_absolute() else root / p
    data_dir, model_dir, output_dir = map(
        location, (args.data_dir, args.model_dir, args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    all_rows = []

    for lane in LANES:
        data_path = data_dir / f'{lane}_val.csv'
        model_path = model_dir / f'{lane}_best_model.joblib'
        if not data_path.exists():
            raise FileNotFoundError(f'Validation CSV missing: {data_path}')
        if not model_path.exists():
            raise FileNotFoundError(f'Model file missing: {model_path}. Run train_tci_models.py')

        df = pd.read_csv(data_path, dtype={'matchId': str})
        if df.empty or 'y_tci' not in df:
            raise ValueError(f'{lane}: validation dataset missing y_tci or empty')
        bundle = joblib.load(model_path)  # Load only model files YOU created/trust.
        if not isinstance(bundle, dict) or not {'pipeline', 'features', 'model_name'} <= bundle.keys():
            raise ValueError(f'{lane}: saved model bundle has unexpected structure')
        model, features = bundle['pipeline'], bundle['features']
        forbidden_tokens = ('gold', 'level', 'xp')
        if any(any(t in col.lower() for t in forbidden_tokens) for col in features):
            raise ValueError(f'{lane}: possible target leakage in feature list')
        missing = [col for col in features if col not in df.columns]
        if missing:
            raise ValueError(f'{lane}: missing features in validation CSV: {missing}')
        # 같은 경기, 같은 시점의 BLUE/RED를 한 쌍으로 정렬
        df = df.sort_values(
            ['matchId', 'minute', 'side']
        ).reset_index(drop=True)

        blue_idx = np.flatnonzero(df['side'].eq('blue').to_numpy())
        red_idx = np.flatnonzero(df['side'].eq('red').to_numpy())

        blue_pairs = df.loc[blue_idx, ['matchId', 'minute']].reset_index(drop=True)
        red_pairs = df.loc[red_idx, ['matchId', 'minute']].reset_index(drop=True)

        if not blue_pairs.equals(red_pairs):
            raise ValueError(f'{lane}: BLUE/RED pair mismatch')
        X = df[features].apply(pd.to_numeric, errors='coerce')
        y = df['y_tci'].to_numpy(dtype=float)
        if X.isna().all().any() or not np.isfinite(y).all():
            raise ValueError(f'{lane}: entirely null feature or invalid targets')

        baseline = rmse(y, model.predict(X))
        rows = []
        for col in features:
            changes = []
            for _ in range(args.repeats):
                shuffled = X.copy()
                # BLUE 값만 섞고 RED는 반대 부호로 설정
                blue_values = X.iloc[blue_idx][col].to_numpy()
                permuted = rng.permutation(blue_values)

                shuffled.iloc[blue_idx, shuffled.columns.get_loc(col)] = permuted
                shuffled.iloc[red_idx, shuffled.columns.get_loc(col)] = -permuted
                changes.append(rmse(y, model.predict(shuffled)) - baseline)
            row = {
                'lane': lane, 'model': bundle['model_name'],
                'feature': col, 'feature_label': display_name(col),
                'importance_rmse_increase': float(np.mean(changes)),
                'importance_std': float(np.std(changes, ddof=1)),
                'baseline_val_rmse': baseline,
                'validation_matches': int(df['matchId'].nunique()),
                'validation_rows': len(df),
            }
            rows.append(row)
        result = pd.DataFrame(rows).sort_values('importance_rmse_increase', ascending=False)
        result.insert(2, 'rank', np.arange(1, len(result) + 1))
        result.to_csv(output_dir / f'{lane}_feature_importance.csv', index=False, encoding='utf-8-sig')
        all_rows.extend(result.to_dict('records'))

        # Feature labels are English for reliable Windows font rendering.
        fig, ax = plt.subplots(figsize=(10, max(4, 0.48 * len(result) + 1.6)))
        ordered = result.iloc[::-1]
        ax.barh(ordered['feature_label'], ordered['importance_rmse_increase'],
                xerr=ordered['importance_std'], capsize=2)
        ax.axvline(0, linestyle='--', linewidth=1)
        ax.set_xlabel('Increase in validation RMSE after feature permutation')
        ax.set_title(f'{lane.upper()} - {bundle["model_name"]} | {df["matchId"].nunique()} validation matches')
        fig.tight_layout()
        fig.savefig(output_dir / f'{lane}_feature_importance.png', dpi=160)
        plt.close(fig)
        top = result.iloc[0]
        print(f'{lane.upper():7} {bundle["model_name"]:17} val RMSE={baseline:.4f} | '
              f'top feature: {top["feature_label"]} ({top["importance_rmse_increase"]:+.4f})')

    pd.DataFrame(all_rows).to_csv(output_dir / 'all_lanes_feature_importance.csv',
                                  index=False, encoding='utf-8-sig')
    print(f'\nDone: CSV and PNG saved to {output_dir}')
    print('Permutation importance shows predictive reliance, not causal impact.')


if __name__ == '__main__':
    main()

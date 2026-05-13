"""
Convert MicroLens-100k to KuaiSim (KuaiRand-Pure) data format.

Interaction mapping rationale:
  - comment presence → is_click=1, is_comment=1, long_view=1
    (comment is stronger signal than view; equivalent to ML-1m rating>3 mapping)
  - all other behavior fields → 0 (not available in microLens)
  - sessions → daily split, identical to KuaiSim's ML-1m adaptation

Output files:
  <output_dir>/log_microlens.csv
  <output_dir>/user_features_microlens.csv
  <output_dir>/video_features_microlens.csv
"""

import argparse
import os

import numpy as np
import pandas as pd


DEFAULT_VIDEO_DURATION_MS = 15_000  # 15s default for micro-videos


def load_interactions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    assert {'user', 'item', 'timestamp'}.issubset(df.columns), (
        "Expected columns: user, item, timestamp"
    )
    df = df.rename(columns={'user': 'user_id', 'item': 'video_id'})
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    return df


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_localize(None)
    df['datetime'] = dt
    df['date'] = dt.dt.strftime('%Y%m%d').astype(int)
    df['hourmin'] = dt.dt.hour * 100 + dt.dt.minute
    df['time_ms'] = (df['timestamp'] * 1000).astype(int)
    return df


def assign_sessions(df: pd.DataFrame) -> pd.DataFrame:
    # One session per (user, day) — same strategy as KuaiSim's ML-1m adaptation
    df['session_id'] = df.groupby(['user_id', 'date']).ngroup()
    df['request_id'] = df.groupby('session_id').cumcount()
    return df


def build_interaction_log(df: pd.DataFrame) -> pd.DataFrame:
    log = pd.DataFrame({
        'user_id':           df['user_id'].values,
        'video_id':          df['video_id'].values,
        'date':              df['date'].values,
        'hourmin':           df['hourmin'].values,
        'time_ms':           df['time_ms'].values,
        # comment = implicit positive: click + comment + long_view all fire
        'is_click':          1,
        'is_like':           0,
        'is_follow':         0,
        'is_comment':        1,
        'is_forward':        0,
        'is_hate':           0,
        'long_view':         1,
        'is_profile_enter':  0,
        'is_rand':           0,
        'play_time_ms':      0,
        'duration_ms':       DEFAULT_VIDEO_DURATION_MS,
        'profile_stay_time': 0,
        'comment_stay_time': 0,
        'tab':               0,
    })
    return log


def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby('user_id').size().rename('n_interactions')
    span = (
        df.groupby('user_id')['timestamp'].max()
        - df.groupby('user_id')['timestamp'].min()
    ) / 86400

    feats = counts.to_frame().join(span.rename('span_days'))

    # user_active_degree: 5-quantile of interaction count (0=least active … 4=most)
    feats['user_active_degree'] = pd.qcut(
        feats['n_interactions'], q=5, labels=[0, 1, 2, 3, 4], duplicates='drop'
    ).astype(int)

    # register_days_range: proxy from activity span quartile
    feats['register_days_range'] = pd.qcut(
        feats['span_days'], q=4, labels=[0, 1, 2, 3], duplicates='drop'
    ).astype(int)

    for col in [
        'is_lowactive_period', 'is_live_streamer', 'is_video_author',
        'follow_user_num_range', 'fans_user_num_range', 'friend_user_num_range',
    ]:
        feats[col] = 0

    for i in range(18):
        feats[f'onehot_feat{i}'] = 0

    feats = feats.reset_index()  # bring user_id back as column

    ordered_cols = (
        ['user_id', 'user_active_degree', 'is_lowactive_period',
         'is_live_streamer', 'is_video_author', 'follow_user_num_range',
         'fans_user_num_range', 'friend_user_num_range', 'register_days_range']
        + [f'onehot_feat{i}' for i in range(18)]
    )
    return feats[ordered_cols]


def build_video_features(
    df: pd.DataFrame, modality_feat_path: str | None = None
) -> pd.DataFrame:
    feats = pd.DataFrame({'video_id': df['video_id'].unique()})

    feats['author_id']      = 0
    feats['video_type']     = 0
    feats['upload_dt']      = 20230101
    feats['upload_type']    = 0
    feats['visible_status'] = 1
    feats['video_duration'] = DEFAULT_VIDEO_DURATION_MS
    feats['server_width']   = 720
    feats['server_height']  = 1280
    feats['music_id']       = 0
    feats['music_type']     = 0
    feats['tag']            = 0

    if modality_feat_path and os.path.exists(modality_feat_path):
        print(f"[video] Loading modality features from {modality_feat_path}")
        modal_df = pd.read_csv(modality_feat_path)
        feat_cols = [c for c in modal_df.columns if c not in ('item_id', 'video_id')]
        id_col = 'item_id' if 'item_id' in modal_df.columns else 'video_id'

        from sklearn.cluster import MiniBatchKMeans
        n_clusters = 100
        print(f"[video] K-means clustering ({n_clusters} clusters) for tag field")
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        modal_df['tag'] = km.fit_predict(modal_df[feat_cols].values)

        feats = feats.merge(
            modal_df[[id_col, 'tag']].rename(columns={id_col: 'video_id', 'tag': 'tag_modal'}),
            on='video_id', how='left',
        )
        feats['tag'] = feats['tag_modal'].fillna(0).astype(int)
        feats = feats.drop(columns=['tag_modal'])
        print(f"[video] Tag assigned to {feats['tag'].ne(0).sum():,} items via modality features")

    ordered_cols = [
        'video_id', 'author_id', 'video_type', 'upload_dt', 'upload_type',
        'visible_status', 'video_duration', 'server_width', 'server_height',
        'music_id', 'music_type', 'tag',
    ]
    return feats[ordered_cols]


def main():
    parser = argparse.ArgumentParser(
        description='Convert MicroLens-100k to KuaiSim (KuaiRand-Pure) format'
    )
    parser.add_argument(
        '--data_path', default='./data/MicroLens-100k_pairs.csv',
        help='Path to MicroLens-100k_pairs.csv',
    )
    parser.add_argument(
        '--modality_feat_path', default=None,
        help='(Optional) Path to extracted_modality_features CSV for tag derivation',
    )
    parser.add_argument(
        '--output_dir', default='./data/MicroLens-KuaiSim',
        help='Output directory for KuaiSim-format files',
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('[1/5] Loading interactions...')
    df = load_interactions(args.data_path)

    print('[2/5] Parsing timestamps and assigning sessions...')
    df = parse_timestamps(df)
    df = assign_sessions(df)

    print('[3/5] Building interaction log...')
    log = build_interaction_log(df)
    log_path = os.path.join(args.output_dir, 'log_microlens.csv')
    log.to_csv(log_path, index=False)
    print(f'       → {log_path}  ({len(log):,} rows)')

    print('[4/5] Building user features...')
    user_feats = build_user_features(df)
    user_path = os.path.join(args.output_dir, 'user_features_microlens.csv')
    user_feats.to_csv(user_path, index=False)
    print(f'       → {user_path}  ({len(user_feats):,} users)')

    print('[5/5] Building video features...')
    video_feats = build_video_features(df, args.modality_feat_path)
    video_path = os.path.join(args.output_dir, 'video_features_microlens.csv')
    video_feats.to_csv(video_path, index=False)
    print(f'       → {video_path}  ({len(video_feats):,} items)')

    print('\n=== Conversion Summary ===')
    print(f'  Users    : {df["user_id"].nunique():,}')
    print(f'  Items    : {df["video_id"].nunique():,}')
    print(f'  Sessions : {df["session_id"].nunique():,}')
    print(f'  Requests : {len(df):,}')
    print(f'  Output   : {args.output_dir}/')


if __name__ == '__main__':
    main()

import argparse
import os

import pandas as pd
from recbole.quick_start import run_recbole


def preprocess(csv_path: str, out_dir: str) -> None:
    """Convert MicroLens-100k pairs CSV to RecBole .inter format."""
    out_file = os.path.join(out_dir, 'microlens100k.inter')
    if os.path.exists(out_file):
        print(f"[preprocess] {out_file} already exists, skipping.")
        return

    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        'user': 'user_id:token',
        'item': 'item_id:token',
        'timestamp': 'timestamp:float',
    })
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_file, sep='\t', index=False)
    print(f"[preprocess] {len(df):,} rows saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(description='Train SASRec on MicroLens-100k')
    parser.add_argument(
        '--data_path',
        default='./data/MicroLens-100k_pairs.csv',
        help='Path to MicroLens-100k_pairs.csv',
    )
    parser.add_argument(
        '--config',
        default='config/sasrec_microlens.yaml',
        help='RecBole config YAML path',
    )
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='Resume training from a .pth checkpoint file',
    )
    args = parser.parse_args()

    preprocess(args.data_path, './dataset/microlens100k')

    config_dict = {}
    if args.checkpoint:
        config_dict['train_from'] = args.checkpoint

    result = run_recbole(
        model='SASRec',
        dataset='microlens100k',
        config_file_list=[args.config],
        config_dict=config_dict or None,
    )

    print('\n' + '=' * 50)
    print('Final Test Result')
    print('=' * 50)
    print(result)


if __name__ == '__main__':
    main()

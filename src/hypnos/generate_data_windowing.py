import math

import joblib
import numpy as np
import yaml
from tqdm import tqdm

from src.hypnos.data_utils import load_data, generate_lbs
from src.hypnos.utils import data_args


def generate_data_windowing(source, win_length=4, stride=0.5):
    """
    Fix win_length and stride (testing first).
    """

    sns, lbs = load_data(source['signal'], source['label'], SEP='\t')

    if sns is None or lbs is None:
        return None, None

    end_idx = min(len(sns) // 1024, len(lbs))
    sns = sns[:end_idx * 1024]
    lbs = lbs[:end_idx]

    sns_new, lbs_pre = [], []

    sns_pad = np.pad(sns, (1024, 1024), mode='constant', constant_values=0)
    lbs_pad = np.pad(lbs, (1, 1), mode='constant', constant_values=-1)

    for i in tqdm(range(8, end_idx * 8 + 8, 1), total=end_idx * 8, desc=f'Prepare Signals & Pre-Label...'):
        s_idx = i - 4
        e_idx = i + 4
        sns_new.append(sns_pad[s_idx * 128: e_idx * 128])
        lb_phs = lbs_pad[math.floor(s_idx * 0.125): math.ceil(e_idx * 0.125)]
        # assume that the window only contains maximum 2 lbs
        if len(lb_phs) == 1:
            lbs_pre.append(([(lb_phs[0], np.float32(4.0))], 0))  # ([(lb_phase, lb_duration)], is_trans_point)
        else:
            trans_point = \
                np.arange(math.ceil(s_idx * 0.125) * 4, math.floor(e_idx * 0.125) * 4 + 1e-9, 4, dtype=np.int32)[0]
            if lb_phs[0] % 3 == lb_phs[1] % 3:
                lbs_pre.append(([(lb_phs[0], np.float32(trans_point - s_idx * 0.5)),
                                 (lb_phs[1], np.float32(e_idx * 0.5 - trans_point))], 0))
            else:
                lbs_pre.append(([(lb_phs[0], np.float32(trans_point - s_idx * 0.5)),
                                 (lb_phs[1], np.float32(e_idx * 0.5 - trans_point))], 1))

    lbs_new = generate_lbs(lbs_pre)

    return np.array(sns_new).astype(np.float32), lbs_new, lbs_pre


if __name__ == '__main__':
    args = data_args()
    config = yaml.load(open(args.data_config, "r"), Loader=yaml.FullLoader)

    raw_data_dir = config['raw_data_dir']
    processed_data_dir = config['processed_data_dir']

    source_ids = config['keymap']

    for source_id in source_ids:
        source = {
            'signal': f'{raw_data_dir}/raw_{source_id}.txt',
            'label': f'{raw_data_dir}/{source_id}.txt',
        }

        sns, lbs, lbs_pre = generate_data_windowing(source)
        print(sns.shape, lbs[0])
        if sns is not None and lbs is not None:
            print(f"Dumping target at {source_id}.pkl and {source_id}_pre.pkl.")
            joblib.dump((sns, lbs), f'{processed_data_dir}/{source_id}.pkl')
            joblib.dump((sns, lbs_pre), f'{processed_data_dir}/{source_id}_pre.pkl')
        else:
            print(f'Chunking interrupt. {source} has no data!')

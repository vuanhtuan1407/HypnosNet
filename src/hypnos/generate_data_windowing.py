import math
from tqdm import tqdm

import numpy as np

from src.hypnos.data_utils import load_data, dump_data, generate_lbs


def generate_data_windowing(source, win_length=4, stride=0.5):
    """
    Fix win_length and stride (testing first).
    """

    sns, lbs = load_data(source)

    if sns is None or lbs is None:
        return None, None

    end_idx = min(len(sns) // 1024, len(lbs))
    sns = sns[:end_idx * 1024]
    lbs = lbs[:end_idx]

    sns_new, lbs_pre = [], []

    sns_pad = np.pad(sns, (1024, 1024), mode='constant', constant_values=0)
    lbs_pad = np.pad(lbs, (1, 1), mode='constant', constant_values=-1)

    for i in tqdm(range(8, end_idx * 8 + 8, 1), total=end_idx * 8, desc=f'Process {source}'):
        s_idx = i - 4
        e_idx = i + 4
        sns_new.append(sns_pad[s_idx * 128: e_idx * 128])
        lb_phs = lbs_pad[math.floor(s_idx * 0.125): math.ceil(e_idx * 0.125)]
        # assume that the window only contains maximum 2 lbs
        if len(lb_phs) == 1:
            lbs_pre.append(([(lb_phs[0], np.float32(4.0))], 0))  # ([(lb_phase, lb_duration)], is_trans_point)
        else:
            trans_point = np.arange(math.ceil(s_idx * 0.125) * 4, math.floor(e_idx * 0.125) * 4 + 1e-9, 4, dtype=np.int32)[0]
            if lb_phs[0] % 3 == lb_phs[1] % 3:
                lbs_pre.append(([(lb_phs[0], np.float32(trans_point - s_idx * 0.5)),
                                 (lb_phs[1], np.float32(e_idx * 0.5 - trans_point))], 0))
            else:
                lbs_pre.append(([(lb_phs[0], np.float32(trans_point - s_idx * 0.5)),
                                 (lb_phs[1], np.float32(e_idx * 0.5 - trans_point))], 1))

    lbs_new = generate_lbs(lbs_pre)

    return np.array(sns_new).astype(np.float32), lbs_new, lbs_pre


if __name__ == '__main__':
    sources = [
        'K3_EEG3_11h',
        'RS2_EEG1_23 hr',
        'S1_EEG1_23 hr'
    ]

    for source in sources:
        sns, lbs, lbs_pre = generate_data_windowing(source)
        print(sns.shape, lbs[0])
        if sns is not None and lbs is not None:
            print(f"Dumping at {source}.pkl and {source}_pre.pkl")
            dump_data(sns, lbs, target=source)
            dump_data(sns, lbs_pre, target=f'{source}_pre')
        else:
            print(f'Chunking interrupt. {source} has no data!')

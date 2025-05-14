import math
from datetime import datetime

import joblib
import numpy as np
from torch.utils.data import Subset
from tqdm import tqdm

from src.hypnos.params import LB_DICT, LB_VEC
from src.hypnos.utils import log


def load_data(signal_file: str, label_file: str, SEP: str = '\t', logger=None):
    dts, sns, lbs = [], [], []

    # read label
    with open(label_file, 'r', encoding='utf-8', errors='replace') as f:
        data = f.readlines()
        start_line = 0
        for i, line in enumerate(data):
            if line.startswith('EpochNo'):
                start_line = i + 1
        for line in tqdm(data[start_line: -1], total=len(data[start_line: -1]), desc=label_file):
            lb = line.split(SEP)[1]
            lb_idx = LB_DICT[lb] if lb in LB_DICT else -1
            lbs.append(lb_idx)

    # read signal
    with open(signal_file, 'r', encoding='utf-8', errors='replace') as f:
        data = f.readlines()
        start_line = 0
        for i, line in enumerate(data):
            if line.startswith('Time'):
                start_line = i + 1
        for line in tqdm(data[start_line: -1], total=len(data[start_line: -1]), desc=signal_file):
            dt, eeg, emg, mot = line.split(SEP)[:4]
            dts.append(dt)
            sns.append([eeg, emg, mot])

    # return np.array(sns), np.array(lbs)

    # check datetime segments
    for i, dt in enumerate(dts[::256]):
        if dt2ms(dt) % 1000 != 0:
            log(f'Error at seg {i}: {dt} ~ {dt2ms(dt)}. Stop chunking data!', logger, 'error')
            return None, None

    # end_idx = min(len(sns) // 1024, len(lbs))
    # sns = sns[:end_idx * 1024]
    # lbs = lbs[:end_idx]

    return np.array(sns).astype(np.float32), np.array(lbs).astype(np.int32)


def chunk_signal(signal_file, label_file, chunk_size=50):
    sns, lbs = load_data(signal_file, label_file)

    if sns is None or lbs is None:
        return None, None

    if len(sns) % (chunk_size * 1024) != 0:
        pad_len = chunk_size * 1024 - len(sns) % (chunk_size * 1024)
        sns = np.pad(sns, (0, pad_len), mode='constant', constant_values=0)

    if len(lbs) % chunk_size != 0:
        pad_len = chunk_size - len(lbs) % chunk_size
        lbs = np.pad(lbs, (0, pad_len), mode='constant', constant_values=-1)

    sns = sns.reshape(-1, chunk_size * 1024)
    lbs = lbs.reshape(-1, chunk_size)

    end_idx = min(len(sns), len(lbs))
    return sns[:end_idx].astype(np.float32), lbs[:end_idx].astype(np.int32)


def segment_data(signal_file, label_file, target_file, SEP='\t', logger=None):
    sns, lbs = load_data(signal_file, label_file, SEP, logger)

    if sns is None or lbs is None:
        return

    end_idx = min(len(sns) // 1024, len(lbs))
    sns = sns[:end_idx * 1024]
    lbs = lbs[:end_idx]

    print((sns.shape[0] / 1024, sns.shape[1]), lbs.shape, lbs[0])
    sns = np.reshape(sns, (-1, 1024, 3))

    data = []
    for sn, lb in zip(sns, lbs):
        lb_vec = LB_VEC[lb]  # init value, NOT final value
        data.append((sn, lb, lb_vec))

    log(f"Dumping target at {target_file}", logger)
    joblib.dump(data, target_file)


def generate_lbs(lbs):
    lbs_new = []
    for (lb_phs, is_trans_point) in lbs:
        lb_ph_new = 0
        for (lb_ph, dur) in lb_phs:
            lb_vector = LB_VEC[lb_ph.item()]
            lb_ph_new = lb_ph_new + np.array(lb_vector).astype(np.float32) * dur.item() / 4
        if is_trans_point == 1:
            lbs_new.append((lb_ph_new, is_trans_point, np.float32(lb_phs[0][1].item())))
        else:
            lbs_new.append((lb_ph_new, is_trans_point, np.float32(0.0)))
    return lbs_new


def generate_data_windowing(signal_file, label_file, target_file, logger=None):
    """
    Win Length = 4s
    Stride = 0.5s
    """

    sns, lbs = load_data(signal_file, label_file, SEP='\t')

    if sns is None or lbs is None:
        return

    end_idx = min(len(sns) // 1024, len(lbs))
    sns = sns[:end_idx * 1024]
    lbs = lbs[:end_idx]

    sns_new, lbs_pre = [], []

    sns_pad = np.pad(sns, (1024, 1024), mode='constant', constant_values=0)
    lbs_pad = np.pad(lbs, (1, 1), mode='constant', constant_values=-1)

    for i in tqdm(range(8, end_idx * 8 + 8, 1), total=end_idx * 8, desc=f'Generating'):
        s_idx = i - 4
        e_idx = i + 4
        sns_new.append(sns_pad[s_idx * 128: e_idx * 128])
        lb_phs = lbs_pad[math.floor(s_idx * 0.125): math.ceil(e_idx * 0.125)]
        # assume that the window only contains a maximum 2 lbs
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
    sns_new = np.array(sns_new).astype(np.float32)

    log(f'Dumping target at {target_file}', logger)
    joblib.dump((sns_new, lbs_new, lbs_pre), target_file)


def generate_data_windowing_v2(signal_file, label_file, target_file, SEP='\t', logger=None):
    sns, lbs = load_data(signal_file, label_file, SEP, logger)

    if sns is None or lbs is None:
        return

    end_idx = min(len(sns) // 1024, len(lbs))
    sns = sns[:end_idx * 1024]
    lbs = lbs[:end_idx]

    print(sns.shape[0] / 1024, sns.shape, lbs.shape, lbs[0])

    sns_pad = np.pad(sns, ((1024, 1024), (0, 0)), mode='constant', constant_values=0)
    lbs_pad = np.pad(lbs, (1, 1), mode='constant', constant_values=-1)

    sns_new, lbs_new, lbs_vec = [], [], []

    for j in tqdm(range(8, end_idx * 8 + 8, 1), total=end_idx * 8, desc=f'Generating'):
        s_idx = j - 4
        e_idx = j + 4
        sns_new.append(sns_pad[s_idx * 128: e_idx * 128])
        lb_phs = lbs_pad[math.floor(s_idx * 0.125): math.ceil(e_idx * 0.125)]
        # assume that the window only contains a maximum 2 lbs (pad_duplicate)
        if len(lb_phs) == 1:
            lbs_new.append(lb_phs[0])
            lbs_vec.append(LB_VEC[lb_phs[0]])
        else:
            lbs_new.append(-1)
            trans_point = \
                np.arange(math.ceil(s_idx * 0.125) * 4, math.floor(e_idx * 0.125) * 4 + 1e-9, 4, dtype=np.int32)[0]
            lb_vec = (
                    np.array(LB_VEC[lb_phs[0]], dtype=np.float32) * (trans_point - s_idx * 0.5)
                    + np.array(LB_VEC[lb_phs[1]], dtype=np.float32) * (e_idx * 0.5 - trans_point)
            )
            lbs_vec.append(lb_vec)

    sns_new = np.array(sns_new).astype(np.float32)
    lbs_new = np.array(lbs_new).astype(np.int32)
    lbs_vec = np.array(lbs_vec).astype(np.float32)

    print(sns_new.shape, lbs_new.shape, lbs_vec.shape)

    data = []
    for sn, lb, lb_vec in zip(sns_new, lbs_new, lbs_vec):
        data.append((sn, lb, lb_vec))

    log(f'Dumping target at {target_file}', logger)
    joblib.dump(data, target_file)


def dt2ms(dt, offset=946659600000, ftype='signal'):
    if ftype == 'signal':
        try:
            return int(datetime.strptime(dt, '%Y.%m.%d.  %H:%M:%S.%f').timestamp()) * 1000 - offset
        except ValueError:
            return int(datetime.strptime(dt, '%m/%d/%Y  %H:%M:%S.%f').timestamp()) * 1000 - offset
    elif ftype == 'label':
        return int(datetime.strptime(dt, '%Y.%m.%d.  %H:%M:%S').timestamp()) * 1000 - offset
    else:
        raise ValueError('ftype must be "signal" or "label"')


def split_train_val_test(dataset, ratio=(0.6, 0.2, 0.2)):
    total_size = len(dataset)
    indices = list(range(len(dataset)))
    train_end = int(ratio[0] * total_size)
    val_end = int((ratio[0] + ratio[1]) * total_size)
    train_ids = indices[:train_end]
    val_ids = indices[train_end:val_end]
    test_ids = indices[val_end:]
    return Subset(dataset, train_ids), Subset(dataset, val_ids), Subset(dataset, test_ids), train_ids, val_ids, test_ids


def split_train_val_test_random(dataset, ratio=(0.6, 0.2, 0.2), seed=42):
    total_size = len(dataset)
    indices = list(range(total_size))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_end = int(ratio[0] * total_size)
    val_end = int((ratio[0] + ratio[1]) * total_size)
    train_ids = indices[:train_end]
    val_ids = indices[train_end:val_end]
    test_ids = indices[val_end:]
    return Subset(dataset, train_ids), Subset(dataset, val_ids), Subset(dataset, test_ids), train_ids, val_ids, test_ids


def split_train_val_random(dataset, ratio=(0.8, 0.2), seed=42):
    total_size = len(dataset)
    indices = list(range(total_size))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_end = int(ratio[0] * total_size)
    train_ids = indices[:train_end]
    val_ids = indices[train_end:]
    return Subset(dataset, train_ids), Subset(dataset, val_ids), train_ids, val_ids


def get_train_val_test_file_random(num_files, seed=42):
    np.random.seed(seed)
    train_files = np.random.choice(num_files, int(num_files * 0.6), replace=False)
    val_files = np.random.choice(num_files, int(num_files * 0.2), replace=False)
    test_files = np.setdiff1d(num_files, np.concatenate([train_files, val_files]))
    return train_files, val_files, test_files


def create_index_mapping(concat_dataset):
    idx_map = {}
    global_idx = 0
    for dt_idx, dt in enumerate(concat_dataset):
        for local_idx in range(len(dt)):
            idx_map[global_idx] = (dt_idx, local_idx)
            global_idx += 1
    return idx_map


if __name__ == "__main__":
    print(generate_lbs([([(np.int32(1), np.float32(2.5)), (np.int32(3), np.float32(1.5))], 1)]))

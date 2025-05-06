from datetime import datetime

import joblib
import numpy as np
from torch.utils.data import Subset
from tqdm import tqdm

from src.hypnos.params import LB_DICT, LB_VEC


def load_data(signal_file: str, label_file: str, SEP: str = '\t'):
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
            print(f'Error at seg {i}: {dt} ~ {dt2ms(dt)}. Stop chunking data!')
            return None, None

    # end_idx = min(len(sns) // 1024, len(lbs))
    # sns = sns[:end_idx * 1024]
    # lbs = lbs[:end_idx]

    return np.array(sns).astype(np.float32), np.array(lbs).astype(np.int32)


def dump_data(sns, lbs, target):
    joblib.dump((sns, lbs), target)


def chunk_signal(source, chunk_size=50):
    sns, lbs = load_data(source)

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


def dt2ms(dt, offset=946659600000, ftype='signal'):
    if ftype == 'signal':
        try:
            return int(datetime.strptime(dt, '%Y.%m.%d.  %H:%M:%S.%f').timestamp()) * 1000 - offset
        except:
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


def split_train_val_test_random(dataset, ratio=(0.6, 0.2, 0.2)):
    total_size = len(dataset)
    indices = list(range(total_size))
    np.random.shuffle(indices)
    train_end = int(ratio[0] * total_size)
    val_end = int((ratio[0] + ratio[1]) * total_size)
    train_ids = indices[:train_end]
    val_ids = indices[train_end:val_end]
    test_ids = indices[val_end:]
    return Subset(dataset, train_ids), Subset(dataset, val_ids), Subset(dataset, test_ids), train_ids, val_ids, test_ids


if __name__ == "__main__":
    print(generate_lbs([([(np.int32(1), np.float32(2.5)), (np.int32(3), np.float32(1.5))], 1)]))

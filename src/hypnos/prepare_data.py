import numpy as np
import yaml

from src.hypnos.logger import get_logger
from src.hypnos.utils import log, parse_data_args
from tests.generate_data_windowing import generate_data_windowing_v2
from tests.segment_data import segment_data


def prepare_data(data_conf, postfix='all', logger=None):
    """
    Prepares and processes data based on the provided configuration.

    This function takes a data configuration and processes it to prepare data for
    further analysis or modeling.

    Notes:
        1. Use the last 2 files for testing.
        2. postfix in ['segment', 'windowing', 'all']
    """

    # Now use postfix == 'all' only
    __prepare_data_all(data_conf, logger)


# def __prepare_data_segment(data_conf, logger=None):
#     raw_data_dir = data_conf['raw_data_dir']
#     processed_data_dir = data_conf['processed_data_dir']
#
#     metainfo = {
#         "train_files": [],
#         "test_files": []
#     }
#
#     for i, (sns_f, lbs_f) in enumerate(zip(data_conf['sns_files'], data_conf['lbs_files'])):
#         if i < len(data_conf['sns_files']) - 2:
#             log("Prepare data for training", logger)
#             segment_data(f"{raw_data_dir}/{sns_f}", f"{raw_data_dir}/{lbs_f}",
#                          f"{processed_data_dir}/train_data_{i}.pkl", logger=logger)
#             metainfo["train_files"].append(f"{processed_data_dir}/train_data_{i}_segment.pkl")
#
#         else:
#             log("Prepare data for testing", logger)
#             segment_data(f"{raw_data_dir}/{sns_f}", f"{raw_data_dir}/{lbs_f}",
#                          f"{processed_data_dir}/test_data_{i}.pkl", logger=logger)
#             metainfo["test_files"].append(f"{processed_data_dir}/test_data_{i}.pkl")
#
#     yaml.dump(metainfo, open(f"{processed_data_dir}/metainfo_segment", "w"))
#     log('Created metainfo_segment.yaml successfully', logger)
#
#
# def __prepare_data_windowing(data_conf, logger=None):
#     raw_data_dir = data_conf['raw_data_dir']
#     processed_data_dir = data_conf['processed_data_dir']
#
#     metainfo = {
#         "train_files": [],
#         "test_files": []
#     }
#
#     for i, (sns_f, lbs_f) in enumerate(zip(data_conf['sns_files'], data_conf['lbs_files'])):
#         if i < len(data_conf['sns_files']) - 2:
#             log("Prepare data for training", logger)
#             generate_data_windowing_v2(f"{raw_data_dir}/{sns_f}", f"{raw_data_dir}/{lbs_f}",
#                                        f"{processed_data_dir}/train_data_{i}.pkl", logger=logger)
#             metainfo["train_files"].append(f"{processed_data_dir}/train_data_{i}_windowing.pkl")
#
#         else:
#             log("Prepare data for testing", logger)
#             segment_data(f"{raw_data_dir}/{sns_f}", f"{raw_data_dir}/{lbs_f}",
#                          f"{processed_data_dir}/test_data_{i}.pkl", logger=logger)
#             metainfo["test_files"].append(f"{processed_data_dir}/test_data_{i}.pkl")
#
#     yaml.dump(metainfo, open(f"{processed_data_dir}/metainfo_windowing", "w"))
#     log('Created metainfo_windowing.yaml successfully', logger)


def __prepare_data_all(data_conf, logger=None):
    raw_data_dir = data_conf['raw_data_dir']
    processed_data_dir = data_conf['processed_data_dir']

    metainfo = {
        "train_files": {
            "segment": [],
            "windowing": []
        },
        "val_files": [],
        "test_files": []
    }

    num_files = min(len(data_conf['sns_files']), len(data_conf['lbs_files']))
    file_ids = np.arange(num_files)
    np.random.seed(42)
    np.random.shuffle(file_ids)

    train_file_ids = file_ids[:-3]
    val_file_ids = file_ids[-3:-2]
    test_file_ids = file_ids[-2:]

    for file_id in train_file_ids:
        i, sns_f, lbs_f = file_id, data_conf['sns_files'][file_id], data_conf['lbs_files'][file_id]
        log("Prepare data for training", logger)
        segment_data(f"{raw_data_dir}/{sns_f}", f"{raw_data_dir}/{lbs_f}",
                     f"{processed_data_dir}/train_data_{i}.pkl", logger=logger)
        metainfo["train_files"]['segment'].append(f"{processed_data_dir}/train_data_{i}_segment.pkl")

        generate_data_windowing_v2(f"{raw_data_dir}/{sns_f}", f"{raw_data_dir}/{lbs_f}",
                                   f"{processed_data_dir}/train_data_{i}.pkl", logger=logger)
        metainfo["train_files"]['windowing'].append(f"{processed_data_dir}/train_data_{i}_windowing.pkl")

    for file_id in val_file_ids:
        i, sns_f, lbs_f = file_id, data_conf['sns_files'][file_id], data_conf['lbs_files'][file_id]
        log("Prepare data for validation", logger)
        segment_data(f"{raw_data_dir}/{sns_f}", f"{raw_data_dir}/{lbs_f}",
                     f"{processed_data_dir}/val_data_{i}.pkl", logger=logger)
        metainfo["val_files"].append(f"{processed_data_dir}/val_data_{i}.pkl")

    for file_id in test_file_ids:
        i, sns_f, lbs_f = file_id, data_conf['sns_files'][file_id], data_conf['lbs_files'][file_id]
        log("Prepare data for testing", logger)
        segment_data(f"{raw_data_dir}/{sns_f}", f"{raw_data_dir}/{lbs_f}",
                     f"{processed_data_dir}/test_data_{i}.pkl", logger=logger)
        metainfo["test_files"].append(f"{processed_data_dir}/test_data_{i}.pkl")

    yaml.dump(metainfo, open(f"{processed_data_dir}/metainfo_segment", "w"))
    log('Created metainfo.yaml successfully', logger)


if __name__ == '__main__':
    data_args = parse_data_args()
    logger = get_logger("DataLogger")
    config = yaml.load(open(data_args.data_config, "r"), Loader=yaml.FullLoader)
    prepare_data(config)

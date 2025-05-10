import yaml

from src.hypnos.logger import get_logger
from src.hypnos.utils import log, parse_data_args
from tests.generate_data_windowing import generate_data_windowing_v2
from tests.segment_data import segment_data


def prepare_data(data_conf, logger=None):
    """
    Prepares and processes data based on the provided configuration.

    This function takes a data configuration and processes it to prepare data for
    further analysis or modeling.

    Notes:
        1. Use the last 2 files for testing.
    """

    raw_data_dir = data_conf['raw_data_dir']
    processed_data_dir = data_conf['processed_data_dir']

    metainfo = {
        "train_files": [],
        "test_files": []
    }

    for i, (sns_f, lbs_f) in enumerate(zip(data_conf['sns_files'], data_conf['lbs_files'])):
        if i < len(data_conf['sns_files']) - 2:
            log("Prepare data for training", logger)
            # prepare for training baseline and testing
            segment_data(f"{raw_data_dir}/{sns_f}", f"{raw_data_dir}/{lbs_f}",
                         f"{processed_data_dir}/train_data_{i}.pkl", logger=logger)
            metainfo["train_files"].append(f"{processed_data_dir}/train_data_{i}.pkl")

            # prepare for training hypnos
            generate_data_windowing_v2(f'{raw_data_dir}/{sns_f}', f'{raw_data_dir}/{lbs_f}',
                                       f'{processed_data_dir}/train_data_{i}_windowing.pkl', logger=logger)
            metainfo["train_files"].append(f"{processed_data_dir}/train_data_{i}_windowing.pkl")
        else:
            log("Prepare data for testing", logger)
            segment_data(f"{raw_data_dir}/{sns_f}", f"{raw_data_dir}/{lbs_f}",
                         f"{processed_data_dir}/test_data_{i}.pkl", logger=logger)
            metainfo["test_files"].append(f"{processed_data_dir}/test_data_{i}.pkl")

    yaml.dump(metainfo, open(f"{processed_data_dir}/metainfo.yaml", "w"))
    log('Created metainfo.yaml successfully', logger)


if __name__ == '__main__':
    data_args = parse_data_args()
    logger = get_logger("DataLogger")
    config = yaml.load(open(data_args.data_config, "r"), Loader=yaml.FullLoader)
    prepare_data(config)

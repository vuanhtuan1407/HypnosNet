import yaml

from src.hypnos.data_utils import segment_data
from src.hypnos.utils import parse_data_args

if __name__ == '__main__':
    args = parse_data_args()
    data_conf = yaml.load(open(args.data_config, "r"), Loader=yaml.FullLoader)
    raw_data_dir = data_conf['raw_data_dir']
    processed_data_dir = data_conf['processed_data_dir']
    for i, (sns_f, lbs_f) in enumerate(zip(data_conf['sns_files'], data_conf['lbs_files'])):
        segment_data(f"{raw_data_dir}/{sns_f}", f"{raw_data_dir}/{lbs_f}",
                     f"{processed_data_dir}/data_{i}_windowing.pkl")

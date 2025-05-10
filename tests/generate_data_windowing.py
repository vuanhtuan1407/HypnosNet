import yaml

from src.hypnos.data_utils import generate_data_windowing_v2
from src.hypnos.utils import parse_data_args

if __name__ == '__main__':
    args = parse_data_args()
    config = yaml.load(open(args.data_config, "r"), Loader=yaml.FullLoader)
    raw_data_dir = config['raw_data_dir']
    processed_data_dir = config['processed_data_dir']
    for i, (sns_f, lbs_f) in enumerate(zip(config['sns_files'], config['lbs_files'])):
        generate_data_windowing_v2(f'{raw_data_dir}/{sns_f}', f'{raw_data_dir}/{lbs_f}',
                                   f'{processed_data_dir}/data_{i}_windowing.pkl')

import joblib
import yaml

from src.hypnos.data_utils import load_data
from src.hypnos.utils import data_args


def process_data(data_conf):
    raw_data_dir = data_conf['raw_data_dir']
    processed_data_dir = data_conf['processed_data_dir']
    for i, (sns_f, lbs_f) in enumerate(zip(data_conf['sns_files'], data_conf['lbs_files'])):
        print(sns_f, lbs_f)
        sns, lbs = load_data(f'{raw_data_dir}/{sns_f}', f'{raw_data_dir}/{lbs_f}')

        end_idx = min(len(sns) // 1024, len(lbs))
        sns = sns[:end_idx * 1024]
        lbs = lbs[:end_idx]

        print((sns.shape[0] / 1024, sns.shape[1]), lbs.shape, lbs[0])

        data = []
        for sn, lb in zip(sns, lbs):
            data.append((sn, lb))

        joblib.dump(data, f'{processed_data_dir}/data_{i}.pkl')


if __name__ == '__main__':
    args = data_args()
    data_conf = yaml.load(open(args.data_config, "r"), Loader=yaml.FullLoader)
    process_data(data_conf)

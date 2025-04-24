import argparse


def training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/conf.yml.example")
    return parser.parse_args()


def data_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str, default="./config/data_conf.yml.example")
    return parser.parse_args()

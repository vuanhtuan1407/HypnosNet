import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/conf.yml.example")
    return parser.parse_args()


def parse_data_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str, default="./config/data_conf.yml.example")
    return parser.parse_args()


def log(msg, logger=None, level="info"):
    if logger is not None:
        if level == "info":
            logger.info(msg)
        elif level == "warning":
            logger.warning(msg)
        elif level == "error":
            logger.error(msg)
        else:
            logger.debug(msg)
    else:
        print(msg)

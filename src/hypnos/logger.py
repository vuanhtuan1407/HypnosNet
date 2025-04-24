import logging
import os
from datetime import datetime


def get_logger(log_dir="logs", log_level=logging.INFO):
    logger = logging.getLogger('HypnosNetLogger')
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


if __name__ == "__main__":
    logger = get_logger(log_dir="../../logs", log_level=logging.DEBUG)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
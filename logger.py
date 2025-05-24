import logging
import os
from datetime import datetime
import csv

def setup_logging():
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    
    log_filename = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(__name__)
    if logger.hasHandlers():
        logger.handlers.clear()

    logging.basicConfig(
        filename=log_filename,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    return logging.getLogger(__name__)

def print_times(n_features, start_point, end_point, train_time, test_time):
    with open("results/windows_times.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([n_features, start_point, end_point, train_time, test_time])
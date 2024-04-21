import glob
import sys
from pathlib import Path
file_path = Path(__file__)
parent_directory_path = file_path.parent.parent
logurupath=str(parent_directory_path)+"\loguru-master"
sys.path.insert(0,logurupath)
from loguru import logger


def log_enable():
    logger.enable("PyALE")
    return None

def log_disable():
    logger.disable("PyALE")
    return None

# logger.add(writer, format="{message}")
# logger.enable(None)
# logger.debug("yes")
# logger.disable(None)
# logger.debug("nope")
stop_the_logger_with_every_files=True
if stop_the_logger_with_every_files==True:
    logger.disable("PyALE")
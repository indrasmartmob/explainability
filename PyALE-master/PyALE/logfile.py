import glob
import sys
from pathlib import Path
file_path = Path(__file__)
parent_directory_path = file_path.parent.parent
logurupath=str(parent_directory_path)+"\loguru-master"
sys.path.insert(0,logurupath)
from loguru import logger
stop_the_logger_with_every_files=False
if stop_the_logger_with_every_files==True:
    logger.disable("PyALE")
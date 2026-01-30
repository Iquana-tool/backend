import logging
import os
from app import create_app
from config import LOGS_DIR
from datetime import datetime

# Set up logging
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOGS_DIR + f"/{datetime.now()}.txt",
    filemode='a',
    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)
app = create_app()


if __name__ == "__main__":
    pass

import logging
import hydra
from app import create_app
from paths import Paths

# Set up logging
logging.basicConfig(
    filename=Paths.logs_dir + "/logs.txt",
    filemode='a',
    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)
app = create_app()


if __name__ == "__main__":
    pass

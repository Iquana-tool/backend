import logging
import hydra
from app import create_app

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = create_app()


if __name__ == "__main__":
    pass

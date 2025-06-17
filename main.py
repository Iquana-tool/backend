import logging
import hydra
from app import create_app

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./app/services/segmentation/configs", config_name="config")
def main():
    app = create_app()


if __name__ == "__main__":
    main()

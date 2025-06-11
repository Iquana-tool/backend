import logging
import argparse
import shutil
import paths

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def delete_db_and_images():
    """Delete the database and all images"""
    logger.info("Deleting database and images...")
    shutil.rmtree(paths.Paths.data_dir)
    logger.info("Database and images deleted.")


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-r', '--reset', action='store_true', help='Reset the database and delete '
                                                                       'all images')
    arg_parser.add_argument('-d', '--dev_mode', action='store_true', help='Run in development mode')
    args = arg_parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.reset:
        delete_db_and_images()
    command = "fastapi.exe dev .\main.py" if args.dev_mode else "fastapi.exe run .\main.py"
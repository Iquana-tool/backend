from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging

from app import create_app

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    app = create_app()

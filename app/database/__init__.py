from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine
from contextlib import contextmanager

# from torch.ao.quantization import qconfig_equals

from config import Paths

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Define the declarative base
database = declarative_base()

# Create an engine
engine = create_engine(Paths.database)

database.metadata.create_all(engine)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    logger.debug("\tInitializing database")
    database.metadata.create_all(bind=engine)


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def get_context_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
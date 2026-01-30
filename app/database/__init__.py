import logging
import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DATABASE_FILE, DATA_DIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Define the declarative general
database = declarative_base()

# Create an engine
if not os.path.exists(DATABASE_FILE):
    os.makedirs(DATA_DIR, exist_ok=True)

engine = create_engine("sqlite:///" + DATABASE_FILE,
                       pool_size=20,  # Default is usually 5
                       max_overflow=50,  # Increase from default 10
                       pool_pre_ping=True,  # Validate connections
                       pool_recycle=3600,  # Recycle after 1 hour
                       )

database.metadata.create_all(engine)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    logger.debug("\tInitializing database")
    database.metadata.create_all(bind=engine)


def get_session():
    session = SessionLocal()
    logging.info(f"DB connections: {engine.pool.checkedout()}")
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_context_session():
    session = SessionLocal()
    logging.info(f"DB connections: {engine.pool.checkedout()}")
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

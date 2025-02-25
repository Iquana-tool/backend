from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine
from contextlib import contextmanager
from config import Paths

# Define the declarative base
database = declarative_base()

# Create an engine
engine = create_engine(Paths.database)

database.metadata.create_all(engine)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Context manager for session handling
@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

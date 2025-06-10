from logging import getLogger
from time import time
from typing import Callable, Any
from functools import wraps

logger = getLogger(__name__)


def log_execution_time(func: Callable) -> Callable:
    """Decorator to log the execution time of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        execution_time = end_time - start_time
        logger.info(f"Execution time for {func.__name__}: {execution_time:.6f} seconds")
        return result

    return wrapper

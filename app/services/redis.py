import redis.asyncio as redis # Ensure you have redis>=4.2.0
from config import REDIS_URL


async def get_redis():
    client = redis.from_url(REDIS_URL, decode_responses=True)
    try:
        yield client
    finally:
        await client.close()

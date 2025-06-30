import asyncio
from redis_manager import RedisSessionManager

async def test_redis():
    manager = RedisSessionManager()
    try:
        await manager.check_connection_health()
        print("Redis connection is healthy.")
    except Exception as e:
        print(f"Redis connection test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_redis())
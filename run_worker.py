import os
from rq import Worker, Queue, Connection
from redis import Redis

# Configure Redis connection
redis_conn = Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=int(os.getenv('REDIS_DB', 0))
)

if __name__ == '__main__':
    # Create a worker
    with Connection(redis_conn):
        q = Queue('transcription')
        worker = Worker([q])
        worker.work() 
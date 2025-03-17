from minio import Minio
import os

HOST = os.getenv('MINIO_HOST')
PORT = os.getenv('MINIO_PORT')
ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
SECURE = os.getenv('MINIO_SECURE') or False
BUCKET_NAME = os.getenv('BUCKET_NAME')


class S3Storage:
    def __init__(self,
                 host=HOST,
                 port=PORT,
                 access_key=ACCESS_KEY,
                 secret_key=SECRET_KEY,
                 secure=SECURE,
                 bucket_name=BUCKET_NAME):
        self.host = host
        self.port = port
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self.bucket_name = bucket_name

    def get_client(self):
        client = Minio(
            f'{self.host}:{self.port}',
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )
        return client

    def create_storage(self):
        client = self.get_client()
        try:
            client.make_bucket(self.bucket_name)
        except Exception as e:
            print(f'Error creating bucket: {e}')

    def new_file(self, object_name, file_name):
        client = self.get_client()
        try:
            client.fput_object(self.bucket_name, object_name, file_name)
        except Exception as e:
            print(f'Error uploading {e}')

    def get(self, transcription_id: int):
        pass

    def append(self, transcription_id: int, new_data: str):
        pass

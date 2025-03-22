
from multi_speaker_transcription import run_transcription
from storage.s3_storage import S3Storage


if __name__ == '__main__':
    storage = S3Storage(host='localhost', port=9000, access_key='ROOTUSER', secret_key='Testpass321', bucket_name='testing')
    run_transcription(audio_file='./audio_files/audio.wav', storage=storage)

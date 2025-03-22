from storage.s3_storage import S3Storage
from pydub import AudioSegment


def load_audio(storage_type: str, audio: str):
    if storage_type == 'S3':
        storage = S3Storage()
        data = storage.get_audio(audio)
    elif storage_type == 'file':
        audio_segments = AudioSegment.from_wav(audio).set_channels(1).set_frame_rate(16000)
        return audio_segments


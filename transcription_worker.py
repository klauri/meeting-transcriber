from rq import Worker, Queue, Connection
from multi_speaker_transcription import Whisper
import os
import numpy as np

# Initialize the Whisper pipeline once when the worker starts
whisper_pipeline = Whisper(whisper_model_dir=f'{os.getenv("MODEL_DIR", "./models")}/{os.getenv("MODEL_NAME", "distil-small.en")}')

def transcribe_segment(audio_segment, start_time, end_time, speaker, storage, transcription_id, audio_file):
    """
    Transcribe a single audio segment and return the result
    """
    raw_data = audio_segment.raw_data
    audio_array = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    result = whisper_pipeline.pipe(audio_array)
    transcription = result['text']
    
    return {
        'start_time': start_time,
        'end_time': end_time,
        'speaker': speaker,
        'transcription': transcription
    }

def process_audio_segment(segment_data):
    """
    Process a single audio segment from the queue
    """
    audio_segment = segment_data['audio_segment']
    start_time = segment_data['start_time']
    end_time = segment_data['end_time']
    speaker = segment_data['speaker']
    storage = segment_data['storage']
    transcription_id = segment_data['transcription_id']
    audio_file = segment_data['audio_file']
    
    return transcribe_segment(
        audio_segment,
        start_time,
        end_time,
        speaker,
        storage,
        transcription_id,
        audio_file
    ) 
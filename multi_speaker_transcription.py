import os
import torchaudio
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.audio import Pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment
import torch
from datetime import timedelta
from math import floor
import numpy as np
from uuid import uuid4
from rq import Queue
from transcription_worker import process_audio_segment
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from whisper_model import Whisper

class TimestampPosition(Enum):
    START = "start"  # Timestamp at the start of the segment
    ACTUAL = "actual"  # Timestamp at the actual speech position

@dataclass
class TranscriptionFormat:
    include_speakers: bool = True
    include_timestamps: bool = True
    timestamp_position: TimestampPosition = TimestampPosition.START
    include_transcription: bool = True
    format_name: str = "full"  # Name for the output file

    @classmethod
    def speakers_only(cls):
        return cls(include_speakers=True, include_timestamps=False, include_transcription=True, format_name="speakers")

    @classmethod
    def timestamps_only(cls):
        return cls(include_speakers=False, include_timestamps=True, include_transcription=True, format_name="timestamps")

    @classmethod
    def transcription_only(cls):
        return cls(include_speakers=False, include_timestamps=False, include_transcription=True, format_name="transcription")

    @classmethod
    def full_format(cls, timestamp_position: TimestampPosition = TimestampPosition.START):
        return cls(include_speakers=True, include_timestamps=True, 
                  timestamp_position=timestamp_position, include_transcription=True,
                  format_name="full")


TRANSCRIPTION_DIR = './transcriptions'
RTTM_DIR = f'{TRANSCRIPTION_DIR}/rttm_files'
AUDIO_DIR = './audio_files'
MODEL_DIR = './models'
MODEL_NAME = 'distil-small.en'

TEMP_DIR = './temp'


def ms_to_hh_mm_ss_str(sec):
    # create timedelta and convert it into string
    td_str = str(timedelta(seconds=sec))
    # split string into individual component
    x = td_str.split(':')
    return f'{x[0]}:{x[1]}:{x[2]}'


class Audio:
    def __init__(self, audio_filename: str, whisper_pipeline, pyannote_model_dir: str, storage, output_formats: Optional[List[TranscriptionFormat]] = None):
        self.transcription_id = uuid4()
        self.audio_file = audio_filename
        self.whisper_pipeline = whisper_pipeline.pipe
        self.whisper_dtype = whisper_pipeline.torch_dtype
        self.pyannote_model = pyannote_model_dir
        self.pyannote_pipeline = Pipeline.from_pretrained(self.pyannote_model)
        self.audio_segments = AudioSegment.from_wav(audio_filename).set_channels(1).set_frame_rate(16000)
        self.storage = storage
        self.queue = Queue('transcription')
        
        # If no formats specified, use full format as default
        self.output_formats = output_formats or [TranscriptionFormat.full_format()]
        
        # Generate keys for each format
        self.transcription_keys = {
            format.format_name: f'{self.transcription_id}-{os.path.basename(self.audio_file)}-{format.format_name}-transcription.txt'
            for format in self.output_formats
        }
        self.rttm_key = f'{self.transcription_id}-{os.path.basename(self.audio_file)}.rttm'

    def split_audio_into_chunks(self, audio_segment, chunk_duration_ms=30000):
        """Split audio into chunks of specified duration."""
        chunks = []
        duration = len(audio_segment)
        for i in range(0, duration, chunk_duration_ms):
            chunk = audio_segment[i:i + chunk_duration_ms]
            chunks.append(chunk)
        return chunks

    def format_transcription_line(self, result, format: TranscriptionFormat):
        """Format a single transcription line based on the output format configuration."""
        start_time = result['start_time']
        end_time = result['end_time']
        speaker = result['speaker']
        transcription = result['transcription']
        
        parts = []
        
        # Add speaker if configured
        if format.include_speakers:
            parts.append(f"{speaker}:")
        
        # Add timestamps if configured
        if format.include_timestamps:
            timestamp = f"[{ms_to_hh_mm_ss_str(start_time)}->{ms_to_hh_mm_ss_str(end_time)}]"
            if format.timestamp_position == TimestampPosition.START:
                parts.insert(0, timestamp)
            else:
                parts.append(timestamp)
        
        # Add transcription (always included)
        parts.append(transcription)
        
        return " ".join(parts) + "\n"

    def process_results(self, results):
        """Process results and generate output for each format."""
        # Sort results by start time
        sorted_results = sorted(results, key=lambda x: x['start_time'])
        
        # Generate output for each format
        for format in self.output_formats:
            transcription_buffer = []
            for result in sorted_results:
                transcription_buffer.append(self.format_transcription_line(result, format))
            
            # Upload the formatted transcription to S3
            self.storage.upload_content(self.transcription_keys[format.format_name], ''.join(transcription_buffer))

    def transcribe(self):
        """Transcribe audio without diarization."""
        # Split audio into 30-second chunks
        chunks = self.split_audio_into_chunks(self.audio_segments)
        
        # Queue each chunk for processing
        jobs = []
        for i, chunk in enumerate(chunks):
            start_time = i * 30  # Each chunk is 30 seconds
            end_time = start_time + 30
            
            job_data = {
                'audio_segment': chunk,
                'start_time': start_time,
                'end_time': end_time,
                'speaker': 'SPEAKER_00',  # Default speaker for non-diarized transcription
                'storage': self.storage,
                'transcription_id': self.transcription_id,
                'audio_file': self.audio_file
            }
            jobs.append(self.queue.enqueue(process_audio_segment, job_data))

        # Wait for all jobs to complete and collect results
        results = []
        for job in jobs:
            result = job.result
            results.append(result)

        # Process results for all formats
        self.process_results(results)

    def diarize(self):
        waveform, sample_rate = torchaudio.load(self.audio_file)

        # run the pipeline on an audio file
        with ProgressHook() as hook:
            diarization = self.pyannote_pipeline({
                "waveform": waveform,
                "sample_rate": sample_rate
                }, hook=hook
            )

        # Get the RTTM content directly
        rttm_content = diarization.to_rttm()
        self.rttm_lines = rttm_content.split('\n')[:-1]

        # Upload RTTM content directly to S3
        self.storage.upload_content(self.rttm_key, rttm_content)

        # Create jobs for each segment
        jobs = []
        
        for line in self.rttm_lines:
            x = line.split(' ')
            start_time = float(x[3])
            duration = float(x[4])
            speaker = x[7]
            end_time = start_time + duration
            audio_segments = []

            if duration < 30.0:
                audio_segments.append(self.audio_segments[start_time*1000:(start_time+duration)*1000])
            else:
                num_splits = duration / 30.0
                offset = duration / num_splits
                new_times = []
                new_end_time = start_time + offset
                for x in range(0, floor(num_splits)):
                    new_times.append([start_time, new_end_time])
                    start_time = end_time
                    new_end_time = start_time + offset

                for new_start, new_end in new_times:
                    audio_segments.append(self.audio_segments[new_start*1000:new_end*1000])

            # Queue each segment for processing
            for audio_seg in audio_segments:
                job_data = {
                    'audio_segment': audio_seg,
                    'start_time': start_time,
                    'end_time': end_time,
                    'speaker': speaker,
                    'storage': self.storage,
                    'transcription_id': self.transcription_id,
                    'audio_file': self.audio_file
                }
                jobs.append(self.queue.enqueue(process_audio_segment, job_data))

        # Wait for all jobs to complete and collect results
        results = []
        for job in jobs:
            result = job.result
            results.append(result)

        # Process results for all formats
        self.process_results(results)


def run_transcription(audio_file: str, storage, use_diarization: bool = False, output_formats: Optional[List[TranscriptionFormat]] = None):
    whisper_pipeline = Whisper(whisper_model_dir=f'{MODEL_DIR}/{MODEL_NAME}')
    audio = Audio(
            audio_filename=audio_file,
            pyannote_model_dir=f"{MODEL_DIR}/config.yaml",
            whisper_pipeline=whisper_pipeline,
            storage=storage,
            output_formats=output_formats)
    
    if use_diarization:
        audio.diarize()
    else:
        audio.transcribe()

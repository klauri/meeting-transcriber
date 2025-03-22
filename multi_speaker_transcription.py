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
    def __init__(self, audio_filename: str, whisper_pipeline, pyannote_model_dir: str, storage):
        self.transcription_id = uuid4()
        self.audio_file = audio_filename
        self.whisper_pipeline = whisper_pipeline.pipe
        self.whisper_dtype = whisper_pipeline.torch_dtype
        self.pyannote_model = pyannote_model_dir
        self.pyannote_pipeline = Pipeline.from_pretrained(self.pyannote_model)
        self.audio_segments = AudioSegment.from_wav(audio_filename).set_channels(1).set_frame_rate(16000)
        self.transcription_key = f'{self.transcription_id}-{os.path.basename(self.audio_file)}-transcription.txt'
        self.rttm_key = f'{self.transcription_id}-{os.path.basename(self.audio_file)}.rttm'
        self.storage = storage
        self.queue = Queue('transcription')

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
        last_speaker = ''
        
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

        # Format and write results
        transcription_buffer = []
        last_speaker = ''
        
        for result in sorted(results, key=lambda x: x['start_time']):
            start_time = result['start_time']
            end_time = result['end_time']
            speaker = result['speaker']
            transcription = result['transcription']

            if speaker == last_speaker:
                output = f'[{ms_to_hh_mm_ss_str(start_time)}->{ms_to_hh_mm_ss_str(end_time)}] {transcription}\n'
            else:
                output = f'{speaker}: [{ms_to_hh_mm_ss_str(start_time)}->{ms_to_hh_mm_ss_str(end_time)}] {transcription}\n\n'

            transcription_buffer.append(output)
            last_speaker = speaker

        # Upload the complete transcription to S3
        self.storage.upload_content(self.transcription_key, ''.join(transcription_buffer))


class Whisper:
    def __init__(self, whisper_model_dir: str):
        self.whisper_model_dir = whisper_model_dir
        self.device = "cuda:0" if torch.cuda.is_available() else torch.float32
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.init_model()

    def init_model(self):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.whisper_model_dir, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )

        processor = AutoProcessor.from_pretrained(self.whisper_model_dir)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            generate_kwargs={"max_new_tokens": 128}
        )

        self.pipe = pipe


def run_transcription(audio_file: str, storage):
    whisper_pipeline = Whisper(whisper_model_dir=f'{MODEL_DIR}/{MODEL_NAME}')
    audio = Audio(
            audio_filename=audio_file,
            pyannote_model_dir=f"{MODEL_DIR}/config.yaml",
            whisper_pipeline=whisper_pipeline,
            storage=storage)
    audio.diarize()

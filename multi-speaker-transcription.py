import os
import torchaudio
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.audio import Pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment
import torch
from datetime import date, timedelta
from math import floor
import numpy as np
from uuid import uuid4


TRANSCRIPTION_DIR = './transcriptions'
AUDIO_DIR = './audio_files'


def ms_to_hh_mm_ss_str(sec):
    # create timedelta and convert it into string
    td_str = str(timedelta(seconds=sec))
    # split string into individual component
    x = td_str.split(':')
    return f'{x[0]}:{x[1]}:{x[2]}'


def audio_prep(audio_file: str):
    audio_formats = torchaudio.utils.ffmpeg_utils.get_audio_decoders()
    print(audio_formats())


class Audio:
    def __init__(self, audio_filename: str, whisper_pipeline, pyannote_model_dir: str):
        self.transcription_id = uuid4()
        self.audio_file = audio_filename
        self.whisper_pipeline = whisper_pipeline.pipe
        self.whisper_dtype = whisper_pipeline.torch_dtype
        self.np_dtype = np.float32 if whisper_pipeline.torch_dtype == torch.float32 else np.float16
        self.pyannote_model = pyannote_model_dir
        self.pyannote_pipeline = Pipeline.from_pretrained(self.pyannote_model)
        self.audio_segments = AudioSegment.from_wav(audio_filename).set_channels(1).set_frame_rate(16000)

    def diarize(self):
        waveform, sample_rate = torchaudio.load(self.audio_file)

        # run the pipeline on an audio file
        with ProgressHook() as hook:
            diarization = self.pyannote_pipeline({
                "waveform": waveform,
                "sample_rate": sample_rate
                }, hook=hook
            )
        # dump the diarization output to disk using RTTM format
        self.rttm_filename = f"{TRANSCRIPTION_DIR}/{self.transcription_id}-{os.path.basename(self.audio_file)}.rttm"
        rttm = diarization.to_rttm()
        self.rttm_lines = rttm.split('\n')[:-1]

        # should do this in separate process
        with open(self.rttm_filename, "w") as rttm:
            diarization.write_rttm(rttm)

        last_speaker = ''
        for line in self.rttm_lines:
            x = line.split(' ')
            start_time = float(x[3])
            duration = float(x[4])
            speaker = x[7]
            end_time = start_time + duration
            audio_seg = []
            transcription = ''
            output = ''

            if duration < 30.0:
                audio_seg.append(self.audio_segments[start_time*1000:(start_time+duration)*1000])
            else:
                num_splits = duration / 30.0
                offset = duration / num_splits
                new_times = []
                new_end_time = start_time + offset
                for x in range(0, floor(num_splits)):
                    new_times.append([start_time, new_end_time])
                    start_time = end_time
                    new_end_time = start_time + offset
                    print(start_time, new_end_time)

                print(new_times)
                for new_start, new_end in new_times:
                    audio_seg.append(self.audio_segments[new_start*1000:new_end*1000])

            for audio in audio_seg:
                raw_data = audio.raw_data
                audio_array = np.frombuffer(raw_data, dtype=np.int16).astype(self.np_dtype) / 32768.0

                result = self.whisper_pipeline(audio_array)
                transcription = transcription + result['text']

            if speaker == last_speaker:
                output = f'[{ms_to_hh_mm_ss_str(start_time)}->{ms_to_hh_mm_ss_str(end_time)}] {transcription}\n'
            else:
                output = f'{speaker}: [{ms_to_hh_mm_ss_str(start_time)}->{ms_to_hh_mm_ss_str(end_time)}] {transcription}\n\n'

            print(output)

            with open(f'{TRANSCRIPTION_DIR}/{self.transcription_id}-{os.path.basename(self.audio_file)}-transcription.txt', 'a') as f:
                f.write(output)

            last_speaker = speaker


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
            max_new_tokens=128,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
        )

        self.pipe = pipe


if __name__ == "__main__":
    audio_prep('./audio_files/audio.wav')
#    whisper_pipeline = Whisper(whisper_model_dir='distil-whisper/distil-small.en')
#    audio = Audio(
#            audio_filename="./audio_files/audio.wav",
#            pyannote_model_dir="./models/config.yaml",
#            whisper_pipeline=whisper_pipeline)
#    audio.diarize()

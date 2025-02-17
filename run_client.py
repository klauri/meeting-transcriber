import torchaudio
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.audio import Pipeline
import whisper
from pydub import AudioSegment
# from whisper_live.client import TranscriptionClient
from datetime import date
import os
from uuid import uuid4


# Load pipeline from disk
pipeline = Pipeline.from_pretrained("models/config.yaml")
model = whisper.load_model('base.en')


def run_pyannote(audio_file: str):
    waveform, sample_rate = torchaudio.load(audio_file)

    # run the pipeline on an audio file
    with ProgressHook() as hook:
        diarization = pipeline({
            "waveform": waveform,
            "sample_rate": sample_rate
            }, hook=hook
        )
    # dump the diarization output to disk using RTTM format
    rttm_filename = f"{date.today()}-audio-output.rttm"
    with open(rttm_filename, "w") as rttm:
        print(diarization)
        diarization.write_rttm(rttm)
    return rttm_filename


def create_segments(rttm_file: str):
    print(f'Creating segments for {rttm_file}')
    with open(rttm_file, 'r') as rttm:
        lines = rttm.readlines()

    last_speaker = ''
    segment_number = 0
    transcription_id = uuid4()

    for line in lines:
        words = line.split(' ')
        print(words)
        time_start = words[3]
        duration = words[4]
        speaker = words[7]
        audio = AudioSegment.from_wav('audio_files/audio.wav')
        segment_start = float(time_start) * 1000
        segment_end = (float(time_start) + float(duration)) * 1000
        print(time_start, segment_start)
        segment = audio[segment_start: segment_end]
        segment.export(f'segments/{segment_number}-segment.wav', format='wav')
        transcription = transcribe_segment(f'segments/{segment_number}-segment.wav')
        with open(f'{transcription_id}-new_transcription.txt', 'a') as file:
            if speaker != last_speaker:
                file.write(f'\n{speaker}: {transcription}')
            else:
                file.write(f'{transcription}')
        last_speaker = speaker

    return segment_number


def transcribe_segment(segment: str):
    result = model.transcribe(segment)
    return result['text']


if __name__ == '__main__':
    rttm_filename = run_pyannote('audio_files\\audio.wav')
    number_of_segments = create_segments('2025-02-04-audio-output.rttm')

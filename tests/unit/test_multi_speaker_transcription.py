import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import torchaudio
from pydub import AudioSegment
import sys
import os
from dotenv import load_dotenv
from redis import Redis
from rq import Connection, Queue
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# Log Redis connection settings
logger.debug("Redis connection settings:")
logger.debug(f"Host: {os.getenv('REDIS_HOST', 'localhost')}")
logger.debug(f"Port: {os.getenv('REDIS_PORT', '6379')}")
logger.debug(f"DB: {os.getenv('REDIS_DB', '0')}")
logger.debug(f"Username: {os.getenv('REDIS_USERNAME', 'taskrunner')}")
logger.debug("Password: [REDACTED]")

# Configure Redis connection
redis_conn = Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=int(os.getenv('REDIS_DB', 0)),
    username=os.getenv('REDIS_USERNAME', 'taskrunner'),
    password=os.getenv('REDIS_PASSWORD', 'msgpass321'),
    decode_responses=True
)

# Test Redis connection
try:
    redis_conn.ping()
    logger.info("Successfully connected to Redis")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {str(e)}")
    raise

# Import the classes after setting up the path
from multi_speaker_transcription import (
    Audio, TranscriptionFormat, TimestampPosition,
    ms_to_hh_mm_ss_str, MODEL_DIR, MODEL_NAME
)
from whisper_model import Whisper

class TestTranscriptionFormat(unittest.TestCase):
    def test_speakers_only_format(self):
        format = TranscriptionFormat.speakers_only()
        self.assertTrue(format.include_speakers)
        self.assertFalse(format.include_timestamps)
        self.assertTrue(format.include_transcription)
        self.assertEqual(format.format_name, "speakers")

    def test_timestamps_only_format(self):
        format = TranscriptionFormat.timestamps_only()
        self.assertFalse(format.include_speakers)
        self.assertTrue(format.include_timestamps)
        self.assertTrue(format.include_transcription)
        self.assertEqual(format.format_name, "timestamps")

    def test_transcription_only_format(self):
        format = TranscriptionFormat.transcription_only()
        self.assertFalse(format.include_speakers)
        self.assertFalse(format.include_timestamps)
        self.assertTrue(format.include_transcription)
        self.assertEqual(format.format_name, "transcription")

    def test_full_format(self):
        format = TranscriptionFormat.full_format()
        self.assertTrue(format.include_speakers)
        self.assertTrue(format.include_timestamps)
        self.assertTrue(format.include_transcription)
        self.assertEqual(format.format_name, "full")
        self.assertEqual(format.timestamp_position, TimestampPosition.START)

class TestTimeConversion(unittest.TestCase):
    def test_ms_to_hh_mm_ss_str(self):
        # Test with 1 hour, 2 minutes, and 3 seconds
        self.assertEqual(ms_to_hh_mm_ss_str(3723), "1:02:03")
        # Test with 0 hours, 1 minute, and 30 seconds
        self.assertEqual(ms_to_hh_mm_ss_str(90), "0:01:30")
        # Test with 0 hours, 0 minutes, and 5 seconds
        self.assertEqual(ms_to_hh_mm_ss_str(5), "0:00:05")

class TestWhisper(unittest.TestCase):
    def test_whisper_initialization_with_local_model(self):
        # Test with actual local model
        whisper = Whisper(os.path.join(MODEL_DIR, MODEL_NAME))
        self.assertIsNotNone(whisper.pipe)
        self.assertIsNotNone(whisper.device)
        self.assertIsNotNone(whisper.torch_dtype)

    @patch('multi_speaker_transcription.AutoModelForSpeechSeq2Seq')
    @patch('multi_speaker_transcription.AutoProcessor')
    @patch('multi_speaker_transcription.pipeline')
    def test_whisper_initialization_cpu(self, mock_pipeline, mock_processor, mock_model):
        # Mock CUDA unavailability
        with patch('torch.cuda.is_available', return_value=False):
            whisper = Whisper(os.path.join(MODEL_DIR, MODEL_NAME))
            self.assertEqual(whisper.device, torch.float32)
            self.assertEqual(whisper.torch_dtype, torch.float32)

class TestAudio(unittest.TestCase):
    def setUp(self):
        self.mock_storage = Mock()
        self.mock_audio_segment = MagicMock(spec=AudioSegment)
        self.mock_audio_segment.__len__.return_value = 60000  # 60 seconds
        # Initialize with actual Whisper model
        self.whisper_pipeline = Whisper(os.path.join(MODEL_DIR, MODEL_NAME))
        # Create a mock queue
        self.mock_queue = Mock(spec=Queue)
        self.mock_queue.name = 'transcription'

    @patch('multi_speaker_transcription.AudioSegment')
    @patch('multi_speaker_transcription.Pipeline')
    @patch('multi_speaker_transcription.Queue')
    def test_audio_initialization(self, mock_queue, mock_pipeline, mock_audio_segment):
        # Mock Queue to return our mock queue
        mock_queue.return_value = self.mock_queue
        # Mock AudioSegment.from_wav to return our mock audio segment
        mock_audio_segment.from_wav.return_value = self.mock_audio_segment
        # Mock os.path.exists to return True for our test file
        with patch('os.path.exists', return_value=True):
            audio = Audio(
                audio_filename="test.wav",
                whisper_pipeline=self.whisper_pipeline,
                pyannote_model_dir=os.path.join(MODEL_DIR, "config.yaml"),
                storage=self.mock_storage
            )
            self.assertIsNotNone(audio.transcription_id)
            self.assertEqual(audio.audio_file, "test.wav")
            self.assertEqual(len(audio.output_formats), 1)
            self.assertEqual(audio.output_formats[0].format_name, "full")

    @patch('multi_speaker_transcription.AudioSegment')
    @patch('multi_speaker_transcription.Queue')
    def test_split_audio_into_chunks(self, mock_queue, mock_audio_segment):
        # Mock Queue to return our mock queue
        mock_queue.return_value = self.mock_queue
        # Mock AudioSegment.from_wav to return our mock audio segment
        mock_audio_segment.from_wav.return_value = self.mock_audio_segment
        # Mock os.path.exists to return True for our test file
        with patch('os.path.exists', return_value=True):
            audio = Audio(
                audio_filename="test.wav",
                whisper_pipeline=self.whisper_pipeline,
                pyannote_model_dir=os.path.join(MODEL_DIR, "config.yaml"),
                storage=self.mock_storage
            )
            chunks = audio.split_audio_into_chunks(self.mock_audio_segment, chunk_duration_ms=30000)
            self.assertEqual(len(chunks), 2)  # 60 seconds split into 30-second chunks

    @patch('multi_speaker_transcription.AudioSegment')
    @patch('multi_speaker_transcription.Queue')
    def test_format_transcription_line(self, mock_queue, mock_audio_segment):
        # Mock Queue to return our mock queue
        mock_queue.return_value = self.mock_queue
        # Mock AudioSegment.from_wav to return our mock audio segment
        mock_audio_segment.from_wav.return_value = self.mock_audio_segment
        # Mock os.path.exists to return True for our test file
        with patch('os.path.exists', return_value=True):
            audio = Audio(
                audio_filename="test.wav",
                whisper_pipeline=self.whisper_pipeline,
                pyannote_model_dir=os.path.join(MODEL_DIR, "config.yaml"),
                storage=self.mock_storage
            )
            result = {
                'start_time': 0.0,
                'end_time': 5.0,
                'speaker': 'SPEAKER_00',
                'transcription': 'Hello world'
            }
            
            # Test full format
            line = audio.format_transcription_line(result, TranscriptionFormat.full_format())
            self.assertIn('SPEAKER_00:', line)
            self.assertIn('[0:00:00->0:00:05]', line)
            self.assertIn('Hello world', line)

            # Test speakers only format
            line = audio.format_transcription_line(result, TranscriptionFormat.speakers_only())
            self.assertIn('SPEAKER_00:', line)
            self.assertNotIn('[', line)
            self.assertIn('Hello world', line)

    @patch('multi_speaker_transcription.torchaudio.load')
    @patch('multi_speaker_transcription.Pipeline')
    @patch('multi_speaker_transcription.AudioSegment')
    @patch('multi_speaker_transcription.Queue')
    def test_diarize(self, mock_queue, mock_audio_segment, mock_pipeline, mock_torchaudio_load):
        # Mock Queue to return our mock queue
        mock_queue.return_value = self.mock_queue
        # Mock AudioSegment.from_wav to return our mock audio segment
        mock_audio_segment.from_wav.return_value = self.mock_audio_segment
        # Mock os.path.exists to return True for our test file
        with patch('os.path.exists', return_value=True):
            # Mock the diarization pipeline
            mock_diarization = Mock()
            # Use proper RTTM format matching audio.rttm
            mock_diarization.to_rttm.return_value = "SPEAKER waveform 1 0.0 2.0 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
            mock_pipeline.from_pretrained.return_value.return_value = mock_diarization

            # Mock torchaudio load
            mock_waveform = torch.randn(1, 16000)
            mock_sample_rate = 16000
            mock_torchaudio_load.return_value = (mock_waveform, mock_sample_rate)

            # Create a mock job with a proper result
            mock_job = Mock()
            mock_job.result = {
                'start_time': 0.0,
                'end_time': 2.0,
                'speaker': 'SPEAKER_00',
                'transcription': 'Test transcription'
            }
            # Make the queue return our mock job
            self.mock_queue.enqueue.return_value = mock_job

            audio = Audio(
                audio_filename="test.wav",
                whisper_pipeline=self.whisper_pipeline,
                pyannote_model_dir=os.path.join(MODEL_DIR, "config.yaml"),
                storage=self.mock_storage
            )

            # Mock the audio segments
            audio.audio_segments = self.mock_audio_segment

            audio.diarize()
            self.mock_storage.upload_content.assert_called()

if __name__ == '__main__':
    # Set up Redis connection for tests
    try:
        with Connection(redis_conn):
            logger.info("Starting tests with Redis connection")
            unittest.main()
    except Exception as e:
        logger.error(f"Error during test execution: {str(e)}")
        raise 
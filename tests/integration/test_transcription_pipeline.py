import unittest
import os
import sys
from pathlib import Path
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load environment variables
load_dotenv()

# Import the classes
from multi_speaker_transcription import (
    Audio, TranscriptionFormat, TimestampPosition,
    ms_to_hh_mm_ss_str, MODEL_DIR, MODEL_NAME
)
from whisper_model import Whisper

class TestTranscriptionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that can be reused across test methods."""
        # Initialize the Whisper model
        cls.whisper_pipeline = Whisper(os.path.join(MODEL_DIR, MODEL_NAME))
        
        # Set up paths
        cls.test_data_dir = Path(__file__).parent.parent / "test_data"
        cls.test_data_dir.mkdir(exist_ok=True)
        
        # Example audio file path
        cls.example_audio_path = cls.test_data_dir / "example.wav"
        
        # Create a mock storage for testing
        class MockStorage:
            def upload_content(self, content, filename):
                logger.debug(f"Mock storage: Uploading {filename}")
                return True
                
        cls.mock_storage = MockStorage()

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Ensure the test data directory exists
        self.test_data_dir.mkdir(exist_ok=True)

    def test_full_transcription_pipeline(self):
        """Test the complete transcription pipeline with the example audio file."""
        # Skip if example audio file doesn't exist
        if not self.example_audio_path.exists():
            self.skipTest("Example audio file not found. Please ensure example.wav exists in the test_data directory.")
            
        # Initialize Audio object
        audio = Audio(
            audio_filename=str(self.example_audio_path),
            whisper_pipeline=self.whisper_pipeline,
            pyannote_model_dir=os.path.join(MODEL_DIR, "config.yaml"),
            storage=self.mock_storage
        )
        
        # Test transcription with different formats
        formats = [
            TranscriptionFormat.full_format(),
            TranscriptionFormat.speakers_only(),
            TranscriptionFormat.timestamps_only(),
            TranscriptionFormat.transcription_only()
        ]
        
        for format in formats:
            with self.subTest(format=format.format_name):
                # Process the audio
                audio.process(format)
                
                # Verify the output
                self.assertIsNotNone(audio.transcription_id)
                self.assertTrue(audio.output_formats)
                
                # Check if the output contains the expected content based on format
                output = audio.get_output(format)
                self.assertIsNotNone(output)
                
                if format.include_speakers:
                    self.assertIn("SPEAKER_", output)
                if format.include_timestamps:
                    self.assertIn("[", output)
                if format.include_transcription:
                    self.assertTrue(len(output.strip()) > 0)

    def test_audio_chunking(self):
        """Test that the audio is properly chunked for processing."""
        if not self.example_audio_path.exists():
            self.skipTest("Example audio file not found.")
            
        audio = Audio(
            audio_filename=str(self.example_audio_path),
            whisper_pipeline=self.whisper_pipeline,
            pyannote_model_dir=os.path.join(MODEL_DIR, "config.yaml"),
            storage=self.mock_storage
        )
        
        # Test with different chunk sizes
        chunk_sizes = [30000, 60000, 90000]  # 30s, 60s, 90s chunks
        
        for chunk_size in chunk_sizes:
            with self.subTest(chunk_size=chunk_size):
                chunks = audio.split_audio_into_chunks(audio.audio_segments, chunk_duration_ms=chunk_size)
                self.assertIsInstance(chunks, list)
                self.assertTrue(len(chunks) > 0)
                
                # Verify each chunk is not longer than the specified duration
                for chunk in chunks:
                    self.assertLessEqual(len(chunk), chunk_size)

    def test_error_handling(self):
        """Test error handling for various edge cases."""
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            Audio(
                audio_filename="nonexistent.wav",
                whisper_pipeline=self.whisper_pipeline,
                pyannote_model_dir=os.path.join(MODEL_DIR, "config.yaml"),
                storage=self.mock_storage
            )
        
        # Test with invalid audio file
        invalid_audio = self.test_data_dir / "invalid.wav"
        invalid_audio.touch()  # Create an empty file
        
        with self.assertRaises(Exception):
            Audio(
                audio_filename=str(invalid_audio),
                whisper_pipeline=self.whisper_pipeline,
                pyannote_model_dir=os.path.join(MODEL_DIR, "config.yaml"),
                storage=self.mock_storage
            )

if __name__ == '__main__':
    unittest.main() 
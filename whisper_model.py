import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class Whisper:
    """
    This class is used to load the Whisper model and the pipeline.
    """
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
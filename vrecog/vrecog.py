from typing import Any
#import torch
#import torchaudio
import whisper
import os
import torch

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

#from pyannote.audio import Pipeline
#import os

import logging
logger = logging.getLogger(__name__)

WHISPER_MODEL=config.WHISPER_MODEL
#WHISPER_MODEL="small"
WHISPER_MODEL_PATH = config.WHISPER_MODEL_PATH
logger.info(f"Loading whisper model: {WHISPER_MODEL}")
model = whisper.load_model(WHISPER_MODEL, download_root = WHISPER_MODEL_PATH)
logger.info(f"Whisper model {WHISPER_MODEL} loaded")



def recognise_text(audio_path: Any) -> str:
    script = model.transcribe(audio_path)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return script["text"] if "text" in script else ""

if __name__ == '__main__':
    print(recognise_text("voices/audio_2024-11-06_18-04-50.ogg"))

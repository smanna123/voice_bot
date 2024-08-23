from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from transformers import WhisperProcessor, WhisperForConditionalGeneration, VitsModel, AutoTokenizer
import torchaudio
import torch
import scipy.io.wavfile
import numpy as np
import io
import os
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.openai import OpenAI
import logging

origins = [
    "https://voice-assistant-react.vercel.app/",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY is not set in the environment variables.")
    raise ValueError("OPENAI_API_KEY is required")

# Setup the models
device = "cpu"
try:
    processor = WhisperProcessor.from_pretrained("whisper_tiny")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("whisper_tiny").to(device)
    vits_model = VitsModel.from_pretrained("tts").to(device)
    tokenizer = AutoTokenizer.from_pretrained("tts")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise HTTPException(status_code=500, detail="Model loading failed")

# Initialize the LLM
llm = OpenAI(temperature=0.5, model="gpt-3.5-turbo")

@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    try:
        # Read audio file
        contents = await file.read()
        audio_stream = io.BytesIO(contents)
        waveform, sample_rate = torchaudio.load(audio_stream)

        # Transcribe audio using Whisper
        input_features = processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_features
        predicted_ids = whisper_model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Generate response using OpenAI
        chat_engine = SimpleChatEngine.from_defaults(llm=llm)
        answer = str(chat_engine.chat(transcription))

        # Synthesize response to audio
        inputs = tokenizer(answer, return_tensors="pt")
        with torch.no_grad():
            output = vits_model(**inputs).waveform
        audio_data = output.squeeze().numpy()
        audio_data = np.int16(audio_data * 32767)

        # Prepare audio data for response
        audio_bytes = io.BytesIO()
        scipy.io.wavfile.write(audio_bytes, rate=vits_model.config.sampling_rate, data=audio_data)
        audio_bytes.seek(0)

        # Return audio stream
        return StreamingResponse(audio_bytes, media_type="audio/wav")
    except Exception as e:
        logger.error(f"An error occurred during audio processing: {e}")
        raise HTTPException(status_code=500, detail="Error processing the audio file")


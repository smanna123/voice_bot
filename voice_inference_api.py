from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from transformers import WhisperProcessor, WhisperForConditionalGeneration, VitsModel, AutoTokenizer
import torchaudio
import torch
import scipy.io.wavfile
import numpy as np
import io
import os
from dotenv import load_dotenv
from openai import OpenAI

app = FastAPI()

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

device = "cpu"
# Load Whisper model and processor
processor = WhisperProcessor.from_pretrained("whisper_tiny")
whisper_model = WhisperForConditionalGeneration.from_pretrained("whisper_tiny").to(device)

# Load Vits model for TTS
vits_model = VitsModel.from_pretrained("tts-test").to(device)
tokenizer = AutoTokenizer.from_pretrained("tts-test")


@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    # Read audio file
    contents = await file.read()
    audio_stream = io.BytesIO(contents)
    waveform, sample_rate = torchaudio.load(audio_stream)

    # Transcribe audio using Whisper
    input_features = processor(waveform.squeeze().numpy(), sampling_rate=sample_rate,
                               return_tensors="pt").input_features
    predicted_ids = whisper_model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # Generate response using OpenAI
    client = OpenAI(api_key=openai_api_key)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": transcription}
        ],
        max_tokens=250
    )
    answer = completion.choices[0].message.content
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


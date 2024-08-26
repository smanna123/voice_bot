from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from transformers import WhisperProcessor, WhisperForConditionalGeneration, VitsModel, AutoModelForCausalLM, AutoTokenizer
import torchaudio
import torch
import scipy.io.wavfile
import numpy as np
import io
import os
from dotenv import load_dotenv
import logging

# Load environment variables and setup logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup CORS and FastAPI app
origins = ["https://voice-assistant-react.vercel.app/"]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize models on CPU
device = torch.device("cpu")
start_token = "assistant"
try:
    processor = WhisperProcessor.from_pretrained("whisper_tiny")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("whisper_tiny").to(device)
    chat_tokenizer = AutoTokenizer.from_pretrained("chat_model")
    chat_model = AutoModelForCausalLM.from_pretrained("chat_model")
    vits_model = VitsModel.from_pretrained("tts").to(device)
    tokenizer = AutoTokenizer.from_pretrained("tts")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise HTTPException(status_code=500, detail="Model loading failed")

@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        audio_stream = io.BytesIO(contents)
        waveform, sample_rate = torchaudio.load(audio_stream)

        # Transcribe audio using Whisper
        input_features = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)
        predicted_ids = whisper_model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Generate response using Chat Model
        messages = [{"role": "user", "content": transcription}]
        input_text = chat_tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = chat_tokenizer.encode(input_text, return_tensors="pt").to(device)
        max_length = inputs.shape[-1] + 300
        outputs = chat_model.generate(
            inputs,
            max_length=max_length,  # Ensure no more than 200 tokens
            temperature=0.3,
            top_p=0.75,
            do_sample=True
        )
        decoded_output = chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
        start_token = "assistant"
        if start_token in decoded_output:
            answer = decoded_output.split(start_token, 1)[-1].strip()
        else:
            answer = decoded_output.strip()

        if len(answer) > 0 and not answer.endswith(('.', '!', '?')):
            last_punct = max(answer.rfind(punct) for punct in ('.', '!', '?'))
            if last_punct != -1:
                answer = answer[:last_punct + 1].strip()

        # Synthesize response to audio using VitsModel
        inputs = tokenizer(answer, return_tensors="pt")
        with torch.no_grad():
            output = vits_model(**inputs).waveform
        audio_data = np.int16(output.squeeze().numpy() * 32767)

        # Prepare audio data for response
        audio_bytes = io.BytesIO()
        scipy.io.wavfile.write(audio_bytes, rate=vits_model.config.sampling_rate, data=audio_data)
        audio_bytes.seek(0)

        return StreamingResponse(audio_bytes, media_type="audio/wav")

    except Exception as e:
        logger.error(f"An error occurred during audio processing: {e}")
        raise HTTPException(status_code=500, detail="Error processing the audio file")

# # Ensure your environment is correctly set to deploy
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

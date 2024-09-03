from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import io
import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.chat_engine import SimpleChatEngine
from gtts import gTTS
import logging

# Set environment variables to maximize CPU usage
os.environ["OMP_NUM_THREADS"] = "16"  # Adjust this as per the number of CPU cores available
os.environ["TF_NUM_INTEROP_THREADS"] = "16"
os.environ["TF_NUM_INTRAOP_THREADS"] = "16"

# Setup CORS, logging, and FastAPI app
origins = ["https://voice-assistant-react.vercel.app/"]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables and validate
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY is not set in the environment variables.")
    raise ValueError("OPENAI_API_KEY is required")

# System prompt for LLM
BROADIFI_WRITING_ASSISTANT = ("You are a Broadifi Voice Assistant powered by the Broadifi AI team. Provide answer of "
                              "user query in a single line and informative and compact, always answer in single line.")

# Setup the models on CPU
device = "cpu"
try:
    processor = WhisperProcessor.from_pretrained("whisper_tiny")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("whisper_tiny").to(device)
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise HTTPException(status_code=500, detail="Model loading failed")

# Initialize the LLM
llm = OpenAI(temperature=0.5, model="gpt-3.5-turbo")


@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    try:
        # Read and process audio file asynchronously
        contents = await file.read()
        audio_stream = io.BytesIO(contents)
        waveform, sample_rate = torchaudio.load(audio_stream)

        # Transcribe audio using Whisper
        input_features = processor(waveform.squeeze().numpy(), sampling_rate=sample_rate,
                                   return_tensors="pt").input_features
        predicted_ids = whisper_model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print(transcription)

        # Generate response using OpenAI
        chat_engine = SimpleChatEngine.from_defaults(system_prompt=BROADIFI_WRITING_ASSISTANT, llm=llm)
        answer = str(chat_engine.chat(transcription))

        # Synthesize response to audio using gTTS
        tts = gTTS(text=answer, lang='en', tld='co.in')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)

        # Return audio stream
        return StreamingResponse(audio_bytes, media_type="audio/wav")
    except Exception as e:
        logger.error(f"An error occurred during audio processing: {e}")
        raise HTTPException(status_code=500, detail="Error processing the audio file")






# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import StreamingResponse
# from starlette.middleware.cors import CORSMiddleware
# from transformers import WhisperProcessor, WhisperForConditionalGeneration, VitsModel, AutoTokenizer
# import torchaudio
# import torch
# import scipy.io.wavfile
# import numpy as np
# import io
# import os
# from dotenv import load_dotenv
# from llama_index.core.chat_engine import SimpleChatEngine
# from llama_index.llms.openai import OpenAI
# import logging
#
# # Set environment variables to maximize CPU usage
# os.environ["OMP_NUM_THREADS"] = "16"  # Adjust this as per the number of CPU cores available
# os.environ["TF_NUM_INTEROP_THREADS"] = "16"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "16"
#
# # Setup CORS, logging, and FastAPI app
# origins = ["https://voice-assistant-react.vercel.app/"]
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"]
# )
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Load environment variables and validate
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
# if not openai_api_key:
#     logger.error("OPENAI_API_KEY is not set in the environment variables.")
#     raise ValueError("OPENAI_API_KEY is required")
#
# # System prompt for LLM
# BROADIFI_WRITING_ASSISTANT = ("You are a Broadifi Voice Assistant powered by the Broadifi AI team. Provide answer of "
#                               "user query in a single line and informative and compact, always answer in single line.")
#
# # Setup the models on CPU
# device = "cpu"
# try:
#     processor = WhisperProcessor.from_pretrained("whisper_tiny")
#     whisper_model = WhisperForConditionalGeneration.from_pretrained("whisper_tiny").to(device)
#     vits_model = VitsModel.from_pretrained("tts").to(device)
#     tokenizer = AutoTokenizer.from_pretrained("tts")
# except Exception as e:
#     logger.error(f"Failed to load models: {e}")
#     raise HTTPException(status_code=500, detail="Model loading failed")
#
# # Initialize the LLM
# llm = OpenAI(temperature=0.5, model="gpt-3.5-turbo")
#
#
# @app.post("/process_audio/")
# async def process_audio(file: UploadFile = File(...)):
#     try:
#         # Read and process audio file asynchronously
#         contents = await file.read()
#         audio_stream = io.BytesIO(contents)
#         waveform, sample_rate = torchaudio.load(audio_stream)
#
#         # Transcribe audio using Whisper
#         input_features = processor(waveform.squeeze().numpy(), sampling_rate=sample_rate,
#                                    return_tensors="pt").input_features
#         predicted_ids = whisper_model.generate(input_features)
#         transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
#         print(transcription)
#
#         # Generate response using OpenAI
#         chat_engine = SimpleChatEngine.from_defaults(system_prompt=BROADIFI_WRITING_ASSISTANT, llm=llm)
#         answer = str(chat_engine.chat(transcription))
#
#         # Synthesize response to audio
#         inputs = tokenizer(answer, return_tensors="pt")
#         with torch.no_grad():
#             output = vits_model(**inputs).waveform
#         audio_data = np.int16(output.squeeze().numpy() * 32767)
#
#         # Prepare audio data for response
#         audio_bytes = io.BytesIO()
#         scipy.io.wavfile.write(audio_bytes, rate=vits_model.config.sampling_rate, data=audio_data)
#         audio_bytes.seek(0)
#
#         # Return audio stream
#         return StreamingResponse(audio_bytes, media_type="audio/wav")
#     except Exception as e:
#         logger.error(f"An error occurred during audio processing: {e}")
#         raise HTTPException(status_code=500, detail="Error processing the audio file")

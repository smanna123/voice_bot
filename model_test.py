# from parler_tts import ParlerTTSForConditionalGeneration
# from transformers import AutoTokenizer, set_seed
# import soundfile as sf
#
# device = "cpu"
#
# model = ParlerTTSForConditionalGeneration.from_pretrained("tts").to(device)
# tokenizer = AutoTokenizer.from_pretrained("tts")
#
#
# prompt = """Bayesian statistics constitute one of the not-so-conventional subareas within statistics, based on a
# particular vision of the concept of probabilities. This post introduces and unveils what bayesian statistics is and
# its differences from frequentist statistics, through a gentle and predominantly non-technical narrative that will
# awaken your curiosity about this fascinating topic.
#
# """
# description = "Elisabeth speaks fast and softly"
#
# input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
# prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
#
# set_seed(42)
# generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
# audio_arr = generation.cpu().numpy().squeeze()
# sf.write("parler_tts_out.wav", audio_arr, 44100)

# from parler_tts import ParlerTTSForConditionalGeneration
# from transformers import AutoTokenizer, set_seed
# import soundfile as sf
# import torch
# from torch.multiprocessing import set_start_method
#
# # Ensure torch multiprocessing starts properly
# try:
#     set_start_method('spawn')
# except RuntimeError:
#     print("failed")
#     pass
#
# # Set number of threads to 8 (for M1 with 8 cores)
# torch.set_num_threads(8)
#
# # Set up the model and tokenizer
# device = "cpu"
# model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-expresso").to(device)
# tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-expresso")
#
# # Define prompt and description
# prompt = """hi, how are you?
# """
#
# description = "Elisabeth speaks fast and softly"
#
# # Tokenize inputs
# input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
# prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
#
# # Set the seed for reproducibility
# set_seed(42)
#
# # Generate speech with greedy sampling mode
# with torch.no_grad():
#     generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
# # num_beam_groups=1
#
# # Convert to numpy array and save as WAV
# audio_arr = generation.cpu().numpy().squeeze()
# sf.write("parler_tts_out.wav", audio_arr, 44100)


# import os
# import time
# import tensorflow as tf
# from TTS.api import TTS
#
# # Set environment variables to maximize CPU usage
# os.environ["OMP_NUM_THREADS"] = "8"  # Adjust this as per the number of CPU cores available
# os.environ["TF_NUM_INTEROP_THREADS"] = "8"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
#
# # Configure TensorFlow to use multiple threads
# config = tf.compat.v1.ConfigProto(
#     intra_op_parallelism_threads=int(os.environ["TF_NUM_INTRAOP_THREADS"]),
#     inter_op_parallelism_threads=int(os.environ["TF_NUM_INTEROP_THREADS"])
# )
# session = tf.compat.v1.Session(config=config)
#
# # Initialize the TTS model
# try:
#     tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
# except Exception as e:
#     print(f"Failed to initialize TTS model: {e}")
#     exit(1)
#
# # Define multiple texts to be converted to speech
# texts_to_speak = [
#     "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
#     "The only thing necessary for the triumph of evil is for good men to do nothing.",
#     "To be yourself in a world that is constantly trying to make you something else is the greatest accomplishment.",
#     "In the end, we will remember not the words of our enemies, but the silence of our friends.",
#     "What lies behind us and what lies before us are tiny matters compared to what lies within us."
# ]
#
# # File path for reference speech file for voice cloning
# reference_speaker_file = "refer.wav"
#
# # Process each text and time the TTS generation
# for i, text in enumerate(texts_to_speak, start=1):
#     output_file_path = f"output_{i}.wav"
#     start_time = time.time()
#     try:
#         tts.tts_to_file(
#             text=text,
#             file_path=output_file_path,
#             speaker_wav=reference_speaker_file,
#             language="en"
#         )
#         end_time = time.time()
#         print(f"Speech generation for text {i} completed successfully in {end_time - start_time:.2f} seconds.")
#     except Exception as e:
#         print(f"Failed to generate or save speech for text {i}: {e}")




import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.openai import OpenAI
import logging
from TTS.api import TTS

# Set environment variables to maximize CPU usage
os.environ["OMP_NUM_THREADS"] = "8"  # Adjust this as per the number of CPU cores available
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"

# Setup CORS and logging
origins = ["https://voice-assistant-react.vercel.app/"]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"]   # Allows all headers
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY is not set in the environment variables.")
    raise ValueError("OPENAI_API_KEY is required")

BROADIFI_WRITING_ASSISTANT = """\
You are a Broadifi Voice Assistant. You are powered by Broadifi AI team. \
You help people come up with creative ideas and content and your answer \
will be very compact and to the point and minimal.
”
"""

# Setup the models
device = "cpu"
try:
    processor = WhisperProcessor.from_pretrained("whisper_tiny")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("whisper_tiny").to(device)
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)  # Text-to-Speech model
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
        chat_engine = SimpleChatEngine.from_defaults(system_prompt=BROADIFI_WRITING_ASSISTANT, llm=llm)
        answer = str(chat_engine.chat(transcription))

        # Synthesize response to audio using the TTS model
        tts_output_path = "response_audio.wav"  # Temporarily store the audio
        tts.tts_to_file(text=answer, file_path=tts_output_path, speaker_wav="refer.wav", language="en")

        # Prepare audio data for response
        with open(tts_output_path, "rb") as audio_file:
            audio_bytes = io.BytesIO(audio_file.read())
        audio_bytes.seek(0)

        # Clean up the temporary file
        os.remove(tts_output_path)

        # Return audio stream
        return StreamingResponse(audio_bytes, media_type="audio/wav")
    except Exception as e:
        logger.error(f"An error occurred during audio processing: {e}")
        raise HTTPException(status_code=500, detail="Error processing the audio file")




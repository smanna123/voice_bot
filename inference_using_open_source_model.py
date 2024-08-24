from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import os

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.to('cpu')
model.config.forced_decoder_ids = None

# Path to the user_voice directory
directory_path = "user_voice"

# Get list of WAV files in the directory
wav_files = [f for f in os.listdir(directory_path) if f.endswith('.wav')]
if not wav_files:
    raise FileNotFoundError("No WAV files found in the directory.")

# Load the first WAV file
file_path = os.path.join(directory_path, wav_files[0])
waveform, sample_rate = torchaudio.load(file_path)

# preprocess the audio
input_features = processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_features

# generate token ids
predicted_ids = model.generate(input_features)

# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(transcription[0])


# pip install transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
checkpoint = "HuggingFaceTB/SmolLM-1.7B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

device = "cpu" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint, quantization_config=quantization_config)

messages = [{"role": "user", "content": transcription[0]}]
input_text=tokenizer.apply_chat_template(messages, tokenize=False)
print(input_text)
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=150, temperature=0.2, top_p=0.9, do_sample=True)
print(tokenizer.decode(outputs[0]))

# from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
# import torch
# import soundfile as sf
# from datasets import load_dataset
#
# processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
# model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
# vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
#
# inputs = processor(text=tokenizer.decode(outputs[0]),
#                    return_tensors="pt")
#
# # load xvector containing speaker's voice characteristics from a dataset
# embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
#
# speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

# sf.write("test.wav", speech.numpy(), samplerate=16000)

# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import StreamingResponse
# from transformers import WhisperProcessor, WhisperForConditionalGeneration, VitsModel, AutoTokenizer
# import torchaudio
# import torch
# import scipy.io.wavfile
# import numpy as np
# import io
# import os
# from dotenv import load_dotenv
# from openai import OpenAI
# from llama_index.core.chat_engine import SimpleChatEngine
# from llama_index.llms.openai import OpenAI
# import logging
#
# app = FastAPI()
#
# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Load environment variables
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
# if not openai_api_key:
#     logger.error("OPENAI_API_KEY is not set in the environment variables.")
#     raise ValueError("OPENAI_API_KEY is required")
#
# # Setup the models
# device = "cpu"
# try:
#     processor = WhisperProcessor.from_pretrained("whisper_tiny")
#     whisper_model = WhisperForConditionalGeneration.from_pretrained("whisper_tiny").to(device)
#     vits_model = VitsModel.from_pretrained("tts-test").to(device)
#     tokenizer = AutoTokenizer.from_pretrained("tts-test")
# except Exception as e:
#     logger.error(f"Failed to load models: {e}")
#     raise HTTPException(status_code=500, detail="Model loading failed")
#
# # Initialize the LLM
# llm = OpenAI(temperature=0.5, model="gpt-3.5-turbo")
#
# @app.post("/process_audio/")
# async def process_audio(file: UploadFile = File(...)):
#     try:
#         # Read audio file
#         contents = await file.read()
#         audio_stream = io.BytesIO(contents)
#         waveform, sample_rate = torchaudio.load(audio_stream)
#
#         # Transcribe audio using Whisper
#         input_features = processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_features
#         predicted_ids = whisper_model.generate(input_features)
#         transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
#
#         # Generate response using OpenAI
#         chat_engine = SimpleChatEngine.from_defaults(llm=llm)
#         answer = str(chat_engine.chat(transcription))
#
#         # Synthesize response to audio
#         inputs = tokenizer(answer, return_tensors="pt")
#         with torch.no_grad():
#             output = vits_model(**inputs).waveform
#         audio_data = output.squeeze().numpy()
#         audio_data = np.int16(audio_data * 32767)
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
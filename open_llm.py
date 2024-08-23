from openai import OpenAI
from transformers import WhisperProcessor, WhisperForConditionalGeneration,VitsModel, AutoTokenizer
import torchaudio
import os
from dotenv import load_dotenv
import torch
import scipy.io.wavfile
import numpy as np


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# load model and processor
processor = WhisperProcessor.from_pretrained("whisper_tiny")
model = WhisperForConditionalGeneration.from_pretrained("whisper_tiny")
#model.to('cuda')
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

# from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": transcription[0]
        }
    ]
)

answer = completion.choices[0].message.content
print(completion.choices[0].message.content)



model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

inputs = tokenizer(answer, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

# Convert waveform from float (-1, 1) to int16 (-32768, 32767)
audio_data = output.squeeze().numpy()
audio_data = np.int16(audio_data * 32767)

# Write to WAV file
scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=audio_data)
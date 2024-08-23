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

device = "cuda" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint, quantization_config=quantization_config)

messages = [{"role": "user", "content": transcription[0]}]
input_text=tokenizer.apply_chat_template(messages, tokenize=False)
print(input_text)
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=150, temperature=0.2, top_p=0.9, do_sample=True)
print(tokenizer.decode(outputs[0]))

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
from datasets import load_dataset

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

inputs = processor(text=tokenizer.decode(outputs[0]),
                   return_tensors="pt")

# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

sf.write("test.wav", speech.numpy(), samplerate=16000)
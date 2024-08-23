from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile
import numpy as np

model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

# Convert waveform from float (-1, 1) to int16 (-32768, 32767)
audio_data = output.squeeze().numpy()
audio_data = np.int16(audio_data * 32767)

# Write to WAV file
scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=audio_data)


from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
from datasets import load_dataset

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
model.to('cpu')
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

inputs = processor(text=completion.choices[0].message.content,
                   return_tensors="pt")

# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

sf.write("test.wav", speech.numpy(), samplerate=18000)
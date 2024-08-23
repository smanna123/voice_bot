# from openai import OpenAI
# client = OpenAI()
#
# completion = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": "Write a haiku about recursion in programming."
#         }
#     ]
# )
#
# print(completion.choices[0].message)

# from transformers import AutoProcessor, AutoModel
# import torch
# import soundfile as sf
#
# processor = AutoProcessor.from_pretrained("suno/bark-small")
# model = AutoModel.from_pretrained("suno/bark-small")
#
# inputs = processor(
#     text=["the capital of france is paris"],
#     return_tensors="pt",
# )
#
# speech_values = model.generate(**inputs, do_sample=True)
# audio = speech_values.squeeze().cpu().numpy()
# sf.write('output_audio.wav', audio, samplerate=16000)

from transformers import AutoProcessor, AutoModel, BarkModel
import torch
import soundfile as sf
from bark import SAMPLE_RATE
from scipy.io.wavfile import write as write_wav

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
# Load the processor and model
processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)
model.enable_cpu_offload()
model =  model.to_bettertransformer()

# Check if a GPU is available and move the model to GPU
voice_preset = "v2/en_speaker_6"
# Process the input text
inputs = processor("the capital of france is paris", voice_preset=voice_preset)


# Move tensors to the same device as the model
inputs = {key: value.to(device) for key, value in inputs.items()}

# Generate the speech
with torch.no_grad():
    speech_values = model.generate(**inputs, do_sample=True)


# Move the tensor back to CPU for saving to file
audio = speech_values.squeeze().cpu().numpy()


sample_rate = model.generation_config.sample_rate
# Save the audio to a WAV file
write_wav("bark_generation.wav", sample_rate, audio)

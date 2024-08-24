from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, set_seed
import soundfile as sf
import torch
import time
from torch.multiprocessing import set_start_method

# Ensure torch multiprocessing starts properly
try:
    set_start_method('spawn')
except RuntimeError:
    print("Multiprocessing start method already set.")
    pass

# Set number of threads to 8 (for M1 with 8 cores)
torch.set_num_threads(8)

# Set up the model and tokenizer
device = "cpu"
model = ParlerTTSForConditionalGeneration.from_pretrained("tts").to(device)
tokenizer = AutoTokenizer.from_pretrained("tts")

# Define prompt and description for a single sentence
prompt = "hi, how are you?"
description = "Elisabeth speaks fast and softly"
output_file = "parler_tts_out.wav"

# Tokenize inputs
input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Set the seed for reproducibility
set_seed(42)

# Start timing
start_time = time.time()

# Generate speech
with torch.no_grad():
    generation = model.generate(
        input_ids=input_ids,
        prompt_input_ids=prompt_input_ids,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        max_length=500,
        num_beams=1
    )

# End timing
end_time = time.time()
time_taken = end_time - start_time
print(f"Time taken: {time_taken:.2f} seconds")

# Convert to numpy array and save as WAV
audio_arr = generation.cpu().numpy().squeeze()
sf.write(output_file, audio_arr, 44100)

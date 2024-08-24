from transformers import VitsModel, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import torch

DIR = "tts"

# model = VitsModel.from_pretrained("facebook/mms-tts-eng")
# tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-expresso")
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-expresso")

model.save_pretrained(DIR)
tokenizer.save_pretrained(DIR)
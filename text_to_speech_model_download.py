from transformers import VitsModel, AutoTokenizer
import torch

DIR = "tts"

model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

model.save_pretrained(DIR)
tokenizer.save_pretrained(DIR)
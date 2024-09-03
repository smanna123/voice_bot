from transformers import WhisperProcessor, WhisperForConditionalGeneration

DIR = "whisper_small-hin"
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
processor.save_pretrained(DIR)
model.save_pretrained(DIR)
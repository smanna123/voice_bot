from transformers import WhisperProcessor, WhisperForConditionalGeneration

DIR = "whisper_tiny"
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
processor.save_pretrained(DIR)
model.save_pretrained(DIR)
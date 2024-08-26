from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

checkpoint = "HuggingFaceTB/SmolLM-1.7B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
DIR = "chat_model"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, quantization_config=quantization_config)
tokenizer.save_pretrained(DIR)
model.save_pretrained(DIR)
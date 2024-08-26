from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from jinja2 import Template
import re

def remove_markdown(text):
    # Remove bold markdown (**text**)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

    # Extend this with other Markdown removal rules if necessary

    return text

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B-Instruct", quantization_config=quantization_config)

# Example messages and template rendering
messages = [
    {"role": "user", "content": "Hello! How are you?"},
    {"role": "assistant", "content": "I'm fine, thank you! And you?"}
]
template = Template("what is MACHINE LEARNING?")
formatted_input = template.render(messages=messages, add_generation_prompt=True)

# Tokenize and generate response
inputs = tokenizer.encode(formatted_input, return_tensors="pt")
outputs = model.generate(inputs, max_length=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

clean_response = remove_markdown(response)
print(clean_response)

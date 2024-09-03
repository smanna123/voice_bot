from transformers import VitsModel, AutoTokenizer
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the Hugging Face access token from the environment variable
hf_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")

DIR = "tts_hin"

# Use the token when loading the model and tokenizer
model = VitsModel.from_pretrained("facebook/mms-tts-hin", use_auth_token=hf_token)
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hin", use_auth_token=hf_token)

# Save the model and tokenizer
model.save_pretrained(DIR)
tokenizer.save_pretrained(DIR)
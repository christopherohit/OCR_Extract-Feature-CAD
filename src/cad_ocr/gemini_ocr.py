import os
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai

load_dotenv('/mnt/d/docker_volume/room_of_nhanhuynh/ai_butler/CAD-ShibaSangyo/full_build_from_scratch_nhanhuynh/.env')


class Gemini_OCR():
    def __init__(self, api_key: str = os.getenv('GOOGLE_API_KEY'), model_name: str = 'gemini-1.5-pro-latest'):
        self.api_key = api_key
        self.model_name = model_name # gemini-pro-vision, gemini-1.5-flash-latest
        self.model = genai.GenerativeModel(self.model_name)

    def loading_data(self, data_batch):
        genai.configure(api_key= self.api_key)

        
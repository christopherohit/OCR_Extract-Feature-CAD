import sys
from src.utils import load_yolo
import random
import google.generativeai as genai

from dotenv import load_dotenv
import glob
import openpyxl

import os
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image
from tqdm import tqdm



class OCR():
    def __init__(self, env_path: str = '', gpf_path: str = '', draw_path: str = '', core_path: str = ''):
        load_dotenv(env_path)
        self.subscription_key = os.environ['AZURE_SHIBA_SUBSCRIPTION_KEY']
        self.endpoint = os.environ['AZURE_SHIBA_ENDPOINT_KEY']
        self.computervision_client = ComputerVisionClient(self.endpoint, CognitiveServicesCredentials(self.subscription_key))
        self.weight_GPF_path = gpf_path
        self.GFP_GAN_Model = None
        self.yolo_draw_model = load_yolo(draw_path)
        self.yolo_core_model = load_yolo(core_path)

        print("Azure Computer Vision Client is ready to use!")
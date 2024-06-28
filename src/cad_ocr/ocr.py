import sys
from src.utils import load_yolo
import random
import time
from dotenv import load_dotenv
import glob
import openpyxl

import os
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image
from tqdm import tqdm
from src.utils import *
from src.utils_azure import *
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import shutil
from gfpgan import GFPGANer
from src.cad_ocr.code_detect import Extract
import pickle

class OCR():
    def __init__(self, env_path: str = '', gpf_path: str = '', draw_path: str = '', core_path: str = '', dict_path: str = '') -> None:
        load_dotenv(env_path)
        self.subscription_key = os.environ['AZURE_SHIBA_SUBSCRIPTION_KEY']
        self.endpoint = os.environ['AZURE_SHIBA_ENDPOINT_KEY']
        self.computervision_client = ComputerVisionClient(self.endpoint, CognitiveServicesCredentials(self.subscription_key))
        self.weight_GPF_path = gpf_path
        self.GFP_GAN_Model = None
        self.yolo_draw_model = load_yolo(draw_path)
<<<<<<< HEAD
        self.yolo_core_model = load_yolo(core_path, type_predict='core')
=======
        self.yolo_core_model = load_yolo(core_path)
>>>>>>> 88a51bf5a255602a6f5f37d3435bfbe3e834db5b
        self.image_object = None
        self.image_full_size = None
        self.path_temp = 'temp'
        self.path_restore_full = None
        self.path_restore_only_cut = None
<<<<<<< HEAD
        self.path_gen_grid = None
=======
>>>>>>> 88a51bf5a255602a6f5f37d3435bfbe3e834db5b
        self.path_save_original = None
        self.path_single_cad_restore = None
        self.result_draw_df = None
        self.result_code_df = None
        self.is_save_single_cad = False
        self.basic_dict = {}
        self.floor_dict = {}
        self.draw_information_dict = {}
        self.workbook =  openpyxl.load_workbook('export.xlsm')
        self.excel =self.workbook['梁欠表3F']
        self.dir_result = 'result/'
        with open(dict_path, 'rb') as f:
            self.dict_class = pickle.load(f)
            print("Load success dict class")
        print("Azure Computer Vision Client is ready to use!") 

    def destroy_all(self) -> None:
<<<<<<< HEAD
        """
        The `destroy_all` function deletes specific directories if they exist and contain files.
        """

=======
>>>>>>> 88a51bf5a255602a6f5f37d3435bfbe3e834db5b
        print('Detect cache')
        print('Deleting all cache')
        self.path_restore_full = os.path.join(self.path_temp, 'temp_azure_cad')
        self.path_restore_only_cut = os.path.join(self.path_temp, 'temp_cad_cut')
        self.path_save_original = os.path.join(self.path_temp, 'temp_original')
        self.path_single_cad = os.path.join(self.path_temp, 'temp_azure_cad_single')
        self.path_single_cad_restore = os.path.join(self.path_temp, 'temp_azure_cad_single_restore')
<<<<<<< HEAD
        self.path_gen_grid = os.path.join(self.path_temp,'temp_gen_grid')
=======
>>>>>>> 88a51bf5a255602a6f5f37d3435bfbe3e834db5b
        if os.path.exists(self.path_restore_full):
            num_file = glob.glob(f'{self.path_restore_full}/*')
            if len(num_file) != 0:
                shutil.rmtree(self.path_restore_full)
        if os.path.exists(self.path_restore_only_cut):
            num_file = glob.glob(f'{self.path_restore_only_cut}/*')
            if len(num_file) != 0:
                shutil.rmtree(self.path_restore_only_cut)
        if os.path.exists(self.path_save_original):
            num_file = glob.glob(f'{self.path_save_original}/*')
            if len(num_file) != 0:
                shutil.rmtree(self.path_save_original)
        if self.is_save_single_cad:
            if os.path.exists(self.path_single_cad):
                num_file = glob.glob(f'{self.path_single_cad}/*')
                if len(num_file) != 0:
                    shutil.rmtree(self.path_single_cad)
<<<<<<< HEAD
        if os.path.exists(self.path_gen_grid):
            num_file = glob.glob(f'{self.path_gen_grid}/*')
            if len(num_file) != 0:
                shutil.rmtree(self.path_gen_grid)
=======
>>>>>>> 88a51bf5a255602a6f5f37d3435bfbe3e834db5b
        if os.path.exists(self.dir_result):
            num_file = glob.glob(f'{self.dir_result}/*')
            if len(num_file) != 0:
                shutil.rmtree(self.dir_result)

    def create_all(self) -> None:
<<<<<<< HEAD
        """
        The function `create_all` creates multiple directories with specified paths if they do not already
        exist.
        """
        
=======
>>>>>>> 88a51bf5a255602a6f5f37d3435bfbe3e834db5b
        os.makedirs(self.path_restore_full, exist_ok= True)
        os.makedirs(self.path_restore_only_cut, exist_ok= True)
        os.makedirs(self.path_save_original, exist_ok= True)
        os.makedirs(self.dir_result, exist_ok= True)
<<<<<<< HEAD
        os.makedirs(self.path_gen_grid, exist_ok= True)
=======
>>>>>>> 88a51bf5a255602a6f5f37d3435bfbe3e834db5b
        if self.is_save_single_cad:
            os.makedirs(self.path_single_cad, exist_ok= True)
            os.makedirs(self.path_single_cad_restore, exist_ok= True)

    # Re-init
    def re_init(self) -> None:
<<<<<<< HEAD
        """
        The `re_init` function initializes multiple attributes to `None` or empty values in a Python class.
        """

        self.path_restore_full = None
        self.path_restore_only_cut = None
        self.path_save_original = None
        self.path_gen_grid = None
=======
        self.path_restore_full = None
        self.path_restore_only_cut = None
        self.path_save_original = None
>>>>>>> 88a51bf5a255602a6f5f37d3435bfbe3e834db5b
        self.image_object = None
        self.result_draw_df = None
        self.image_full_size = None
        self.raw_code_df_full = pd.DataFrame(columns=['boxes','cls','angle'])
        self.path_single_cad = None
        self.path_single_cad_restore = None
        self.basic_dict = {}
        self.floor_dict = {}
        self.draw_information_dict = {}
        self.dir_result = 'result/'

    def save_pil_to_file(self,pil_image: Image) -> str:
<<<<<<< HEAD
        """
        The function `save_pil_to_file` saves a PIL image to a file and returns the path to the saved file.
        
        Args:
            pil_image (Image) : The `pil_image` parameter in the `save_pil_to_file` method is expected to be an
            instance of the `Image` class. This parameter represents the PIL (Python Imaging Library) image that
            you want to save to a file

        Return: 
            The function `save_pil_to_file` returns the path where the PIL image is saved as a PNG file.
        """
=======
>>>>>>> 88a51bf5a255602a6f5f37d3435bfbe3e834db5b
        path_to_save = os.path.join(self.path_save_original,'temp_original_file.png')
        pil_image.save(path_to_save)
        return path_to_save

    def __config_gfpgan__(self):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
                                    scale=2,
                                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                                    model=model,
                                    tile=400,
                                    tile_pad=10,
                                    pre_pad=0,
                                    half=True)
        self.GFP_GAN_Model = GFPGANer(model_path=self.weight_GPF_path,
                                        upscale=2,
                                        arch='clean',
                                        channel_multiplier=2,
                                        bg_upsampler=bg_upsampler)
        
        print("GFPGAN Model is ready to use ...!")
    
    def upscale_image(self, image_path: str = '') -> Image:
        img_high_resolution = upscale_image(image_path= image_path,
                                            restorer= self.GFP_GAN_Model,
                                            des_path=self.path_restore_full)
        img_high_resolution = Image.open(img_high_resolution)
        return img_high_resolution
    
    def extract_draw(self, image_high_resolution:Image):
        def get_left(bbox):
            return bbox[0]
        image_high_resolution_extract = image_high_resolution.copy()
        result_draw_df = yolo_inference(image_high_resolution_extract,
                                        self.yolo_draw_model,
                                        type= 'draw')
        
        result_draw_df['xmin'] = result_draw_df['boxes'].apply(get_left)
        self.result_draw_df = result_draw_df.sort_values(by='xmin').drop(columns='xmin').reset_index(drop=True)
        # if self.is_save_single_cad:
        #     count = 0
        #     for temp in self.result_draw_df.boxes:
        #         x1,y1,x2,y2 = temp
        #         image_specfied_cad = image_high_resolution_extract.crop((x1,y1,x2,y2))
        #         name_specified_cad = f'image_specified_{x1}_{y1}_{x2}_{y2}.png'
        #         image_specfied_cad.save(os.path.join(self.path_single_cad,name_specified_cad))

        #         image_specfied_restore = upscale_image()
        
    def extract_code(self, image_high_resolution:Image):
        image_high_resolution_extract = image_high_resolution.copy()
<<<<<<< HEAD
        self.result_code_df = yolo_inference(image_high_resolution_extract,
=======
        self.raw_code_df = yolo_inference(image_high_resolution_extract,
>>>>>>> 88a51bf5a255602a6f5f37d3435bfbe3e834db5b
                                               self.yolo_core_model,
                                               type= 'code')
    
    def processOCR(self, image_high_res: Image)->tuple[Dict,Dict]:
        draw_information_list = {}
        base_image_pil, draw_location_list = split_images(image_high_res.copy(), self.result_draw_df)
        base_info = get_basic_info(base_image_pil, self.computervision_client)
<<<<<<< HEAD
        print(draw_location_list)
        for idx, draw_location in enumerate(draw_location_list):
            floor_str = str(idx)
            code_df = self.result_code_df[self.result_code_df.boxes.apply(lambda b: calc_ovl(b, draw_location) > 0.5)]
            code_df.reset_index(drop=True, inplace=True)
            print(len(code_df['boxes']))
            num_cell, cell_width, cell_height = get_cell_size(code_df)
            num_cell = int(num_cell)
            concat_image_pil = gen_grid_image(image_high_res, code_df, num_cell, cell_width, cell_height)
            concat_image_pil.save(os.path.join(self.path_gen_grid,f'gen_{idx}.png'))
=======
        for idx, draw_location in enumerate(draw_location_list):
            floor_str = str(idx)
            code_df = self.raw_code_df[self.raw_code_df.boxes.apply(lambda b: calc_ovl(b, draw_location) > 0.4)]
            code_df.reset_index(drop=True, inplace=True)
            num_cell, cell_width, cell_height = get_cell_size(code_df)
            num_cell = int(num_cell)
            concat_image_pil = gen_grid_image(image_high_res, code_df, num_cell, cell_width, cell_height)
>>>>>>> 88a51bf5a255602a6f5f37d3435bfbe3e834db5b
            draw_words, draw_boxes = ocr_azure_cad(concat_image_pil.copy(), self.computervision_client)
            extract_dict = mapping_code(image_high_res, draw_words, draw_boxes, code_df,
                                        num_cell, cell_width, cell_height)
            extract_dict = combine_rephrase_dict_2(extract_dict)
            draw_information_list[int(floor_str)] = extract_dict
<<<<<<< HEAD
            time.sleep(2)
=======
>>>>>>> 88a51bf5a255602a6f5f37d3435bfbe3e834db5b
        return draw_information_list, base_info
    
    def calculator_similar(self, draw_information_list:Dict|List):
        """
        The function `calculator_similar` takes a list of dictionaries as input, checks for similarity with
        existing codes in a dictionary, and returns a list of corresponding codes.
        
        Args:
            draw_information_list (Dict|List) : The `draw_information_list` parameter is expected to be a dictionary
            or a list containing information about drawings. Each element in the list represents a drawing, and
            each drawing is a dictionary where the keys are indices and the values are codes
        
        Return: 
            The function `calculator_similar` returns a dictionary where each key corresponds to an
            index in the `draw_information_list` and the value is a list of results after processing the
            information in the input list.
        """
        list_infor_cad = {}
        for idx_cad in range(len(draw_information_list)):
            list_infor_cad[idx_cad] = []
<<<<<<< HEAD
            for idx_core, core in draw_information_list[idx_cad].items():
                if core == '':
                    core = ''
=======
            for idx_core, core in draw_information_list[idx_cad+1].items():
                if core == '':
                    core = 'A'
>>>>>>> 88a51bf5a255602a6f5f37d3435bfbe3e834db5b
                
                if core in self.dict_class:
                    result = core
                else:
                    list_score_similarity = []
                    print('The system not found this code this dict so we will calculate similarity of it with all code in dict')
                    for idx_core_cad in self.dict_class:
                        score_similarity = calculator_similarity_str(str_1=core, str_2=idx_core_cad)
                        list_score_similarity.append(score_similarity)
                    idx_core_max = list_score_similarity.index(max(list_score_similarity))
                    result = self.dict_class[idx_core_max]
                list_infor_cad[idx_cad].append(result)
        return list_infor_cad
    
<<<<<<< HEAD
    def clear_all_excel(self)->None:
        print("="*12,"Clean Cache in Excel File","="*12)
        self.excel['E3'] = ''
        self.excel['E4'] = ''
        self.excel['B3'] = ''
        self.excel['G3'] = ''
        start_from = 8
        to_end = 57
        for i in range(start_from, to_end+1):
            self.excel[f'C{i}'] = ''
            self.excel[f'D{i}'] = ''
            self.excel[f'O{i}'] = ''
            self.excel[f'P{i}'] = ''
            self.excel[f'AB{i}'] = ''
            self.excel[f'AC{i}'] = ''
            self.workbook.save('export.xlsm')

    def write_to_excel(self, draw_information_list: Dict = {}, base_info: Dict = {})->str:
        
        base_info = preprocessing_str(basic_info_dict= base_info)
        try:
            self.excel['E3'] = base_info['construct_type']
        except:
            self.excel['E3'] = base_info['construct_type'][0]
=======
    def write_to_excel(self, draw_information_list: Dict = {}, base_info: Dict = {})->str:
        base_info = preprocessing_str(basic_info_dict= base_info)
        self.excel['E3'] = base_info['construct_type'][0]
>>>>>>> 88a51bf5a255602a6f5f37d3435bfbe3e834db5b
        self.excel['E4'] = base_info['num_S']
        self.excel['B3'] = base_info['builder_name'][0]
        self.excel['G3'] = base_info['anken'][0]
        print(draw_information_list)
        for idx_cad in tqdm(range(len(draw_information_list))):
            start_from = 8
            if idx_cad == 0:
<<<<<<< HEAD
                for core_fill in draw_information_list[idx_cad]:
=======
                for idx,core_fill in draw_information_list[idx_cad].items():
>>>>>>> 88a51bf5a255602a6f5f37d3435bfbe3e834db5b
                    self.excel[f'C{start_from}'] = core_fill
                    if core_fill.startswith('K') or core_fill.startswith('M') or core_fill.startswith('SH'):
                        self.excel[f'D{start_from}'] = '上\n下'
                    start_from = start_from + 1
                self.workbook.save(os.path.join(self.dir_result, 'result.xlsx'))
            elif idx_cad == 1:
<<<<<<< HEAD
                for core_fill in draw_information_list[idx_cad]:
                    self.excel[f'O{start_from}'] = core_fill
                    if core_fill.startswith('K') or core_fill.startswith('M') or core_fill.startswith('SH'):
                        self.excel[f'P{start_from}'] = '上\n下'
                    start_from = start_from + 1
                self.workbook.save(os.path.join(self.dir_result, 'result.xlsx'))
            elif idx_cad == 2:
                for core_fill in draw_information_list[idx_cad]:
                    self.excel[f'AB{start_from}'] = core_fill
                    if core_fill.startswith('K') or core_fill.startswith('M') or core_fill.startswith('SH'):
                        self.excel[f'AC{start_from}'] = '上\n下'
                    start_from = start_from + 1
=======
                for idx,core_fill in draw_information_list[idx_cad].items():
                    self.excel[f'O{start_from}'] = core_fill
                    if core_fill.startswith('K') or core_fill.startswith('M') or core_fill.startswith('SH'):
                        self.excel[f'P{start_from}'] = '上\n下'
                        start_from = start_from + 1
                self.workbook.save(os.path.join(self.dir_result, 'result.xlsx'))
            elif idx_cad == 2:
                for idx,core_fill in draw_information_list[idx_cad].items():
                    self.excel[f'AB{start_from}'] = core_fill
                    if core_fill.startswith('K') or core_fill.startswith('M') or core_fill.startswith('SH'):
                        self.excel[f'AC{start_from}'] = '上\n下'
                        start_from = start_from + 1
>>>>>>> 88a51bf5a255602a6f5f37d3435bfbe3e834db5b
                self.workbook.save(os.path.join(self.dir_result, 'result.xlsx'))
        # self.workbook.save(os.path.join(self.dir_result, 'result.xlsx'))
        return os.path.join(self.dir_result, 'result.xlsx')


from src.utils import load_yolo
import time
from dotenv import load_dotenv
import glob
import openpyxl

import os
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image
from tqdm import tqdm
from src.utils import *
from src.utils_azure import *
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import shutil
from gfpgan import GFPGANer
import pickle
import torch

class OCR():
    def __init__(self, env_path: str = '', gpf_path: str = '', draw_path: str = '', core_path: str = '', dict_path: str = '') -> None:
        load_dotenv(env_path)
        self.subscription_key = os.environ['AZURE_SHIBA_SUBSCRIPTION_KEY']
        self.endpoint = os.environ['AZURE_SHIBA_ENDPOINT_KEY']
        self.computervision_client = ComputerVisionClient(self.endpoint, CognitiveServicesCredentials(self.subscription_key))
        self.weight_GPF_path = gpf_path
        self.GFP_GAN_Model = None
        self.paddle_model = PaddleOCR(ocr_version="PP-OCRv4",
                            lang = 'en')
        self.yolo_draw_model = load_yolo(draw_path)
        self.yolo_core_model = load_yolo(core_path, type_predict= 'core')
        self.model_parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
        self.image_object = None
        self.image_full_size = None
        self.path_temp = 'temp'
        self.path_restore_full = None
        self.path_restore_only_cut = None
        self.path_gen_grid = None
        self.path_save_original = None
        self.path_save_inform = None
        self.path_save_core= None
        self.path_single_cad_restore = None
        self.path_save_core_restore = None
        self.result_draw_df = None
        self.result_code_df = None
        self.path_save_num = None
        self.path_save_char = None
        self.is_save_single_cad = False
        self.basic_dict = {}
        self.floor_dict = {}
        self.draw_information_dict = {}
        self.workbook =  openpyxl.load_workbook('export.xlsm')
        self.excel =self.workbook['梁欠表3F']
        self.dir_result = 'result/'
        self.is_double_restore = False
        with open(dict_path, 'rb') as f:
            self.dict_class = pickle.load(f)
            print("Load success dict class")
        print("Azure Computer Vision Client is ready to use!")
        self.path_temp_cad_1 = None
        self.path_temp_cad_2 = None
        self.path_temp_cad_3 = None
        self.path_temp_cad_1_restore = None
        self.path_temp_cad_2_restore = None
        self.path_temp_cad_3_restore = None
        self.path_temp_text_detect = None

    def destroy_all(self) -> None:
        print('Detect cache')
        print('Deleting all cache')
        self.path_restore_full = os.path.join(self.path_temp, 'temp_azure_cad')
        self.path_restore_only_cut = os.path.join(self.path_temp, 'temp_cad_cut')
        self.path_save_original = os.path.join(self.path_temp, 'temp_original')
        self.path_single_cad = os.path.join(self.path_temp, 'temp_azure_cad_single')
        self.path_single_cad_restore = os.path.join(self.path_temp, 'temp_azure_cad_single_restore')
        self.path_gen_grid = os.path.join(self.path_temp,'temp_gen_grid')
        self.path_save_inform = os.path.join(self.path_temp, 'temp_cad_inform')
        self.path_save_core = os.path.join(self.path_temp, 'temp_paddle_paddle')
        self.path_save_core_restore = os.path.join(self.path_temp, 'temp_save_core_restore')
        self.path_save_num = os.path.join(self.path_temp, 'temp_num')
        self.path_save_char = os.path.join(self.path_temp, 'temp_char')
        self.path_temp_cad_1 = os.path.join(self.path_save_core, '0')
        self.path_temp_cad_2 = os.path.join(self.path_save_core, '1')
        self.path_temp_cad_3 = os.path.join(self.path_save_core, '2')
        self.path_temp_cad_1_restore = os.path.join(self.path_save_core_restore, '0')
        self.path_temp_cad_2_restore = os.path.join(self.path_save_core_restore, '1')
        self.path_temp_cad_3_restore = os.path.join(self.path_save_core_restore, '2')
        self.path_temp_text_detect = os.path.join(self.path_temp, 'temp_text_detect')
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
        if os.path.exists(self.path_gen_grid):
            num_file = glob.glob(f'{self.path_gen_grid}/*')
            if len(num_file) != 0:
                shutil.rmtree(self.path_gen_grid)
        if os.path.exists(self.path_save_inform):
            num_file = glob.glob(f'{self.path_save_inform}/*')
            if len(num_file) != 0:
                shutil.rmtree(self.path_save_inform)
        if os.path.exists(self.path_save_core):
            num_file = glob.glob(f'{self.path_save_core}/*')
            if len(num_file) != 0:
                shutil.rmtree(self.path_save_core)

        if os.path.exists(self.path_save_core_restore):
            num_file = glob.glob(f'{self.path_save_core_restore}/*')
            if len(num_file) != 0:
                shutil.rmtree(self.path_save_core_restore)
        
        if os.path.exists(self.path_save_num):
            num_file = glob.glob(f'{self.path_save_num}/*')
            if len(num_file) != 0:
                shutil.rmtree(self.path_save_num)
        
        if os.path.exists(self.path_save_char):
            num_file = glob.glob(f'{self.path_save_char}/*')
            if len(num_file) != 0:
                shutil.rmtree(self.path_save_char)

        if os.path.exists(self.dir_result):
            num_file = glob.glob(f'{self.dir_result}/*')
            if len(num_file) != 0:
                shutil.rmtree(self.dir_result)

        if os.path.exists(self.path_temp_cad_1):
            num_file = glob.glob(f'{self.path_temp_cad_1}/*')
            if len(num_file) != 0:
                shutil.rmtree(self.path_temp_cad_1)
        if os.path.exists(self.path_temp_cad_2):
            num_file = glob.glob(f'{self.path_temp_cad_2}/*')
            if len(num_file) != 0:
                shutil.rmtree(self.path_temp_cad_2)
        if os.path.exists(self.path_temp_cad_3):
            num_file = glob.glob(f'{self.path_temp_cad_3}/*')
            if len(num_file) != 0:
                shutil.rmtree(self.path_temp_cad_3)
        if os.path.exists(self.path_temp_cad_1_restore):
            num_file = glob.glob(f'{self.path_temp_cad_1_restore}/*')
            if len(num_file) != 0:
                shutil.rmtree(self.path_temp_cad_1_restore)
        if os.path.exists(self.path_temp_cad_2_restore):
            num_file = glob.glob(f'{self.path_temp_cad_2_restore}/*')
            if len(num_file) != 0:
                shutil.rmtree(self.path_temp_cad_2_restore)
        if os.path.exists(self.path_temp_cad_3_restore):
            num_file = glob.glob(f'{self.path_temp_cad_3_restore}/*')
            if len(num_file) != 0:
                shutil.rmtree(self.path_temp_cad_3_restore)
        if os.path.exists(self.path_temp_text_detect):
            num_file = glob.glob(f'{self.path_temp_text_detect}/*')
            if len(num_file) != 0:
                shutil.rmtree(self.path_temp_text_detect)

    def create_all(self) -> None:
        os.makedirs(self.path_restore_full, exist_ok= True)
        os.makedirs(self.path_restore_only_cut, exist_ok= True)
        os.makedirs(self.path_save_original, exist_ok= True)
        os.makedirs(self.dir_result, exist_ok= True)
        os.makedirs(self.path_gen_grid, exist_ok= True)
        os.makedirs(self.path_save_inform, exist_ok= True)
        os.makedirs(self.path_save_core, exist_ok= True)
        os.makedirs(self.path_save_core_restore, exist_ok= True)
        os.makedirs(self.path_save_num, exist_ok= True)
        os.makedirs(self.path_save_char, exist_ok= True)
        os.makedirs(self.path_temp_text_detect, exist_ok= True)
        if self.is_save_single_cad:
            os.makedirs(self.path_single_cad, exist_ok= True)
            os.makedirs(self.path_single_cad_restore, exist_ok= True)

    # Re-init
    def re_init(self) -> None:
        self.path_restore_full = None
        self.path_restore_only_cut = None
        self.path_save_original = None
        self.path_gen_grid = None
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
        self.path_save_inform = None
        self.path_save_core = None
        self.path_save_core_restore = None
        self.path_save_num = None
        self.path_save_char = None
        self.path_temp_cad_1_restore = None
        self.path_temp_cad_1 = None
        self.path_temp_cad_2 = None
        self.path_temp_cad_2_restore = None
        self.path_temp_cad_3 = None
        self.path_temp_cad_3_restore = None
        self.path_temp_text_detect = None

    def save_pil_to_file(self,pil_image: Image) -> str:
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
        def get_bottom(bbox):
            return bbox[1]
        image_high_resolution_extract = image_high_resolution.copy()
        result_draw_df = yolo_inference(image_high_resolution_extract,
                                        self.yolo_draw_model,
                                        type= 'draw')
        
        result_draw_df['xmin'] = result_draw_df['boxes'].apply(get_left)
        self.result_draw_df = result_draw_df.sort_values(by='xmin').drop(columns='xmin').reset_index(drop=True)
        if len(self.result_draw_df['boxes']) > 2:
            position = []
            for i in range(len(self.result_draw_df['boxes'])):
                if len(position) == 2 and 1 not in position:
                    position.append(1)
                    break
                elif len(position) == 2 and 2 not in position:
                    position.append(2)
                    break
                elif len(position) == 2 and 3 not in position:
                    position.append(3)
                    break
                else:
                    if abs(get_left(self.result_draw_df['boxes'][i]) - get_left(self.result_draw_df['boxes'][i+1])) > 400:
                        if get_left(self.result_draw_df['boxes'][i]) < get_left(self.result_draw_df['boxes'][i+1]):
                            position.append(2)
                        else:
                            position.append(3)
                    else:
                        if get_bottom(self.result_draw_df['boxes'][i]) < get_bottom(self.result_draw_df['boxes'][i]):
                            position.append(2)
                        else:
                            position.append(1)
            self.result_draw_df['position'] = position
            self.result_draw_df = self.result_draw_df.sort_values(by='position').drop(columns='position').reset_index(drop= True)


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
        result_code_df = yolo_inference(image_high_resolution_extract,
                                               self.yolo_core_model,
                                               type= 'code')
        return result_code_df
    
    def processOCR(self,image, image_high_res: Image)->tuple[Dict,Dict]:
        draw_information_list = {}
        base_image_pil, _ = split_images(image.copy(), self.result_draw_df)
        base_image_pil.save(f"{self.path_save_inform}/temp.png")
        base_info = get_basic_info(base_image_pil, self.computervision_client)
        self.extract_draw(image_high_res)
        _, draw_location_list = split_images(image_high_res.copy(), self.result_draw_df)
        
        for idx, draw_location in enumerate(draw_location_list):

            floor_str = str(idx)
            if int(floor_str) == 0:
                path_save = self.path_temp_cad_1
                path_save_restore = self.path_temp_cad_1_restore
                os.makedirs(self.path_temp_cad_1, exist_ok= True)
                os.makedirs(self.path_temp_cad_1_restore, exist_ok= True)
                os.makedirs(f"{self.path_save_num}/{floor_str}", exist_ok= True)
                os.makedirs(f"{self.path_save_char}/{floor_str}", exist_ok= True)
                os.makedirs(f'{self.path_temp_text_detect}/{floor_str}', exist_ok= True)
                self.path_save_num = os.path.join(self.path_save_num, floor_str)
                self.path_save_char = os.path.join(self.path_save_char, floor_str)
                self.path_temp_text_detect = os.path.join(self.path_temp_text_detect, floor_str)
                
            elif int(floor_str) == 1:
                self.path_save_num = 'temp/temp_num/'
                self.path_save_char = 'temp/temp_char/'
                self.path_temp_text_detect = 'temp/temp_text_detect/'
                path_save = self.path_temp_cad_2
                path_save_restore = self.path_temp_cad_2_restore
                os.makedirs(self.path_temp_cad_2, exist_ok= True)
                os.makedirs(self.path_temp_cad_2_restore, exist_ok= True)
                os.makedirs(f"{self.path_save_num}/{floor_str}", exist_ok= True)
                os.makedirs(f"{self.path_save_char}/{floor_str}", exist_ok= True)
                os.makedirs(f'{self.path_temp_text_detect}/{floor_str}', exist_ok= True)
                self.path_save_num = os.path.join(self.path_save_num, floor_str)
                self.path_save_char = os.path.join(self.path_save_char, floor_str)
                self.path_temp_text_detect = os.path.join(self.path_temp_text_detect, floor_str)
                
            elif int(floor_str) == 2:
                self.path_save_num = 'temp/temp_num/'
                self.path_save_char = 'temp/temp_char/'
                self.path_temp_text_detect = 'temp/temp_text_detect/'
                path_save = self.path_temp_cad_3
                path_save_restore = self.path_temp_cad_3_restore
                os.makedirs(self.path_temp_cad_3, exist_ok= True)
                os.makedirs(self.path_temp_cad_3_restore, exist_ok= True)
                os.makedirs(f"{self.path_save_num}/{floor_str}", exist_ok= True)
                os.makedirs(f"{self.path_save_char}/{floor_str}", exist_ok= True)
                os.makedirs(f'{self.path_temp_text_detect}/{floor_str}', exist_ok= True)
                self.path_save_num = os.path.join(self.path_save_num, floor_str)
                self.path_save_char = os.path.join(self.path_save_char, floor_str)
                self.path_temp_text_detect = os.path.join(self.path_temp_text_detect, floor_str)
                
            
            code_img = image_high_res.crop(draw_location)
            code_img.save(f'{self.path_restore_only_cut}/{floor_str}.png')
            print('Done almost')
            code_df = yolo_inference(code_img,
                                     self.yolo_core_model,
                                     type= 'code')
            # code_df = self.result_code_df[self.result_code_df.boxes.apply(lambda b: calc_ovl(b, draw_location) > 0.5)]
            # code_df.reset_index(drop=True, inplace=True)
            num_cell, cell_width, cell_height = get_cell_size(code_df)
            num_cell = int(num_cell)
            code_df = reparse_CAD(code_df, draw_location)
            concat_image_pil = gen_grid_image(image_high_res, code_df, num_cell, cell_width, cell_height, is_saved= True, path_save= path_save)
            concat_image_pil.save(os.path.join(self.path_gen_grid,f'gen_{idx}.png'))

            
            list_image_word = []
            list_image_num = []
            if self.is_double_restore:
                for image_path in glob.glob(f'{path_save}/*'):
                    upscale_image(image_path= image_path, restorer= self.GFP_GAN_Model, is_saved=True, des_path= path_save_restore)
                
                for i in glob.glob(f'{path_save_restore}/*'):
                    name_file = os.path.basename(i)
                    basename, _ = os.path.splitext(name_file)
                    image = Image.open(i)
                    path_num, path_char = detect_circle_new(image_pil= image,
                                                        name= basename,
                                                        path_num= self.path_save_num,
                                                        path_char= self.path_save_char)
                    if path_num == 'Error' and path_char == 'Error':
                        continue
                    else:
                        list_image_num.append(path_num)
                        list_image_word.append(path_char)
            else:
                for i in glob.glob(f'{path_save}/*'):
                    name_file = os.path.basename(i)
                    basename, _ = os.path.splitext(name_file)
                    image = Image.open(i)
                    path_num, path_char = detect_circle_new(image_pil= image,
                                                          name= basename,
                                                          path_num= self.path_save_num,
                                                          path_char= self.path_save_char)
                    if path_num == 'Error' and path_char == 'Error':
                        continue
                    else:
                        list_image_num.append(path_num)
                        list_image_word.append(path_char)
            list_char = []
            list_num = []
            
            for index in range(len(list_image_word)):
                word, _ = text_detect(list_image_word[index], self.path_temp_text_detect, model = self.paddle_model, parseq_model= self.model_parseq)
                if word == 'non-circle':
                    continue
                else:                    
                    image_num = Image.open(list_image_num[index])
                    num = extract_output(self.model_parseq, image_num)
                    try:
                        num = int(num)
                        if num >= 48:
                            num = text_recognition(image_num)
                        list_num.append(int(num))
                        list_char.append(word)
                    except:
                        try:
                            num = text_recognition(image_num)
                            list_num.append(int(num))
                            list_char.append(word)
                        except:
                            continue
            extract_dict = {'digit_code': list_num,
                            'code': list_char}
            # print(type(concat_image_pil))
            # draw_words, draw_boxes = ocr_azure_cad(concat_image_pil.copy(), self.computervision_client)
            # extract_dict = mapping_code(image_high_res, draw_words, draw_boxes, code_df,
            #                             num_cell, cell_width, cell_height)
            extract_dict = combine_rephrase_dict_2(extract_dict)
            draw_information_list[int(floor_str)] = extract_dict
            
            # draw_words, draw_boxes = ocr_azure_cad(concat_image_pil.copy(), self.computervision_client)
            # extract_dict = mapping_code(image_high_res, draw_words, draw_boxes, code_df,
            #                             num_cell, cell_width, cell_height)
            # extract_dict = combine_rephrase_dict_2(extract_dict)
            # draw_information_list[int(floor_str)] = extract_dict
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
            for idx_core, core in draw_information_list[idx_cad].items():
                if core == '':
                    core = 'A'
                
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

    def write_to_excel(self, draw_information_list: Dict = {}, base_info: Dict = {}, using_base_dict: bool = False)->str:
        
        base_info = preprocessing_str(basic_info_dict= base_info)
        try:
            self.excel['E3'] = base_info['construct_type']
        except:
            self.excel['E3'] = base_info['construct_type'][0]
        self.excel['E4'] = base_info['num_S']
        self.excel['B3'] = base_info['builder_name'][0]
        self.excel['G3'] = base_info['anken'][0]
        print(draw_information_list)
        if using_base_dict:
            for idx_cad in tqdm(range(len(draw_information_list))):
                start_from = 8
                if idx_cad == 0:
                    for core_fill in draw_information_list[idx_cad]:
                        self.excel[f'C{start_from}'] = core_fill
                        if core_fill.startswith('K') or core_fill.startswith('M') or core_fill.startswith('SH'):
                            self.excel[f'D{start_from}'] = '上\n下'
                        start_from = start_from + 1
                    self.workbook.save(os.path.join(self.dir_result, 'result.xlsx'))
                elif idx_cad == 1:
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
                    self.workbook.save(os.path.join(self.dir_result, 'result.xlsx'))
            # self.workbook.save(os.path.join(self.dir_result, 'result.xlsx'))
            return os.path.join(self.dir_result, 'result.xlsx')
        else:
            for idx_cad in tqdm(range(len(draw_information_list))):
                start_from = 8
                if idx_cad == 0:
                    for idxcore,core_fill in draw_information_list[idx_cad].items():
                        self.excel[f'C{start_from}'] = core_fill
                        if core_fill.startswith('K') or core_fill.startswith('M') or core_fill.startswith('SH'):
                            self.excel[f'D{start_from}'] = '上\n下'
                        start_from = start_from + 1
                    self.workbook.save(os.path.join(self.dir_result, 'result.xlsx'))
                elif idx_cad == 1:
                    for idxcore, core_fill in draw_information_list[idx_cad].items():
                        self.excel[f'O{start_from}'] = core_fill
                        if core_fill.startswith('K') or core_fill.startswith('M') or core_fill.startswith('SH'):
                            self.excel[f'P{start_from}'] = '上\n下'
                        start_from = start_from + 1
                    self.workbook.save(os.path.join(self.dir_result, 'result.xlsx'))
                elif idx_cad == 2:
                    for idxcore, core_fill in draw_information_list[idx_cad].items():
                        self.excel[f'AB{start_from}'] = core_fill
                        if core_fill.startswith('K') or core_fill.startswith('M') or core_fill.startswith('SH'):
                            self.excel[f'AC{start_from}'] = '上\n下'
                        start_from = start_from + 1
                    self.workbook.save(os.path.join(self.dir_result, 'result.xlsx'))
            # self.workbook.save(os.path.join(self.dir_result, 'result.xlsx'))
            return os.path.join(self.dir_result, 'result.xlsx')
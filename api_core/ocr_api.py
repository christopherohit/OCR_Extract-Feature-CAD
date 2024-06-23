import sys
sys.path.append('../')
import time
import os
from src.utils import load_image
from fastapi.responses import FileResponse
from src.cad_ocr.ocr import OCR
import shutil
from pdf2image import convert_from_path

main_process = OCR(env_path='.env', gpf_path= 'GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth',
                   draw_path='weight/best_draw.pt', core_path= 'weight/best_code.pt', dict_path = 'weight/dict.pkl')

def check_image(file)-> bool:
    if type(file) == 'PIL.JpegImagePlugin.JpegImageFile' or type(file) == 'PIL.PngImagePlugin.PngImageFile' or type(file):
        return True
    return False

def convert_pdf_to_image(dpi = 200, file_path = ''):
    images = convert_from_path(file_path, dpi=dpi)
    return images[0]


async def process(file):

    if len(os.listdir('request_file/')) != 0:
        shutil.rmtree('request_file/')
        os.makedirs('request_file/', exist_ok=True)
    with open('request_file/cache.png', "wb") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
    file = load_image('request_file/cache.png')
    
    # print('Check data type of file')
    # if check_image(file) != True:
    #     print('File is not an image')
    #     print('Converting from PDF to Image')
    #     file_cache = main_process.save_pil_to_file(pil_image= file)

    #     print('Save PIL to cache file done')
    #     file = convert_pdf_to_image(file)

    print('Starting config CAD')
    main_process.destroy_all()
    main_process.create_all()
    main_process.__config_gfpgan__()

    print('Save PIL to cache file')
    file_cache = main_process.save_pil_to_file(pil_image= file)
    print('Save PIL to cache file done')
    
    print('Start Image resolution')
    image_high_resolution = main_process.upscale_image(image_path= file_cache)
    
    print('Extract CAD and Core of it')
    main_process.extract_draw(image_high_resolution= image_high_resolution)
    main_process.extract_code(image_high_resolution= image_high_resolution)

    print("Progressing OCR ...")
    try:
        all_char, basic_inform = main_process.processOCR(image_high_res= image_high_resolution)
    except:
        print("Error in OCR...\nTry to sleep 3s before recall")
        time.sleep(3)
        all_char, basic_inform = main_process.processOCR(image_high_res= image_high_resolution)

    print('Benmark comparing result to dict class ...')
    # result_comparing = main_process.calculator_similar(draw_information_list= all_char)

    print('Writing result to excel file...')
    path_result =  main_process.write_to_excel(all_char, basic_inform)
    return FileResponse(path_result, filename='result.xlsx')
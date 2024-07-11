from src.cad_ocr.ocr_3 import OCR
from src.utils import load_image
from fastapi.responses import FileResponse
import glob
import time

main_process = OCR(env_path='.env', gpf_path= 'GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth',
                   draw_path='weight/best_draw.pt', core_path= 'weight/v10x_1280_finetunning.pt', dict_path = 'weight/dict.pkl')

file_image = glob.glob('assets/*.png')
for file in file_image:
    image = load_image(file)
    print('Starting config CAD')
    main_process.destroy_all()
    main_process.create_all()
    main_process.__config_gfpgan__()
    print('Save PIL to cache file')
    file_cache = main_process.save_pil_to_file(pil_image= image)
    print('Save PIL to cache file done')
    print('Start Image resolution')
    image_high_resolution = main_process.upscale_image(image_path= file_cache)
    print('Extract CAD and Core of it')
    main_process.extract_draw(image_high_resolution= image)
    # list_code = main_process.extract_code(image_high_resolution= image)

    print("Progressing OCR ...")
    # try:
    all_char, basic_inform = main_process.processOCR(image , image_high_res= image_high_resolution)
    # except:
    #     print("Error in OCR...\nTry to sleep 3s before recall")
    #     time.sleep(3)
    #     all_char, basic_inform = main_process.processOCR(image,image_high_res= image_high_resolution)
    
    save_result = 'cache/'
    name = f"{file.split('/')[-1].split('.')[0]}.xlsx"
    result_comparing = main_process.calculator_similar(draw_information_list= all_char)
    print('Writing result to excel file...')
    main_process.dir_result = save_result
    path_result =  main_process.write_to_excel(result_comparing, basic_inform, using_base_dict= True, name= name)
    main_process.dir_result = 'result/'
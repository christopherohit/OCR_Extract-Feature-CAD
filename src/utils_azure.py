from PIL import Image
import io
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from typing import Dict, List, Tuple
from rapidfuzz import process, fuzz
import pandas as pd
import os
from src.utils import *
from msrest.authentication import CognitiveServicesCredentials
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

def get_basic_info(image_pil: Image, computervision_client: ComputerVisionClient) -> Dict:
    """
    The function `get_basic_info` extracts basic information from a document image using Azure Cognitive
    Services OCR and returns the extracted data in a dictionary format.
    
    Args:
        image_pil (Image) : The `image_pil` parameter is a Pillow image object that contains the basic
        information of a document. It is used as input to extract text information using OCR (Optical
        Character Recognition) methods
 
        computervision_client (ComputerVisionClient) : The `computervision_client` parameter is likely an instance of the
        ComputerVisionClient class, which is used for interacting with the Azure Cognitive Services Computer
        Vision API. This client would be responsible for making requests to the Computer Vision API for
        tasks such as optical character recognition (OCR) on images to extract

    Return: 
        The function `get_basic_info` returns a dictionary containing the following keys and
        values:
        - "construct_type": a list containing the extracted construct type string
        - "builder_name": a list containing the extracted builder name string
        - "anken": a list containing the extracted anken string
        - "ocr": a dictionary with keys "words" and "boxes" containing the OCR results (words and
    """
    """Get basic information of document

    Args:
        image_pil (Image): pillow image of basic information

    Returns:
        Dict: The basic information 
    """
    words, boxes = ocr_azure_cad(image_pil.copy(), computervision_client)
    data_df = pd.DataFrame({'words': words, 'boxes': boxes})
    data_df[['left', 'top', 'right', 'bottom']] = data_df['boxes'].apply(pd.Series)
    data_df['words'] = data_df['words'].apply(lambda e: e.replace(' ', ''))

    try:
        construct_type = data_df[data_df.words.str.contains('ids', case=False)]
        construct_type_str = construct_type.words.tolist()[0]
    except:
        construct_type_str = ''

    try:
        # 分 譲 住 宅 建 築 販 売 設 計 施 工
        title_builder_name = process.extractOne("分 譲 住 宅 建 築 販 売 設 計 施 工", words, scorer=fuzz.WRatio)
        title_builder_name_idx = title_builder_name[2]
        title_builder_name_box = boxes[title_builder_name_idx]
        title_builder_name_width = (title_builder_name_box[2] - title_builder_name_box[0]) * 0.5
        
        match_builder_name = data_df[(data_df.left < (title_builder_name_box[2] + title_builder_name_width)) & (data_df.right > title_builder_name_box[0]) &
                                    (data_df.top > title_builder_name_box[1])]
        match_builder_name.sort_values(by='top', inplace=True)
        match_builder_name = match_builder_name[(match_builder_name.iloc[0].top < match_builder_name.bottom) & (match_builder_name.iloc[0].bottom > match_builder_name.top)]
        match_builder_name_str = ' '.join(match_builder_name.sort_values(by='left').words.tolist())
    except:
        match_builder_name_str = ''

    try:
        anken_title = process.extractOne("TITLE", words, scorer=fuzz.WRatio)
        title_anken_idx = anken_title[2]
        title_anken_box = boxes[title_anken_idx]
        title_anken_width = (title_anken_box[2] - title_anken_box[0]) * 4
        match_anken = data_df[(data_df.left < (title_anken_box[2] + title_anken_width)) & 
                            (data_df.right > title_anken_box[0]) &
                            (data_df.top > title_anken_box[1])]
        match_anken.sort_values(by='top', inplace=True)
        match_anken = match_anken[(match_anken.iloc[0].top < match_anken.bottom) & (match_anken.iloc[0].bottom > match_anken.top)]
        match_anken_str = ' '.join(match_anken.sort_values(by='left').words.tolist())
    except:
        match_anken_str = ''
    
    try:
        number_extract = process.extractOne('号棟', words, scorer= fuzz.WRatio)
        num_idx = number_extract[2]
        num_box = boxes[num_idx]
        _, second_nearest_bbox = find_nearest_and_second_nearest_bbox((num_box[0],num_box[1]), boxes)
        get_index_second_nearest_bbox = boxes.index(second_nearest_bbox)
        match_number = int(words[get_index_second_nearest_bbox])
    except:
        match_number = ''
    
    return {"construct_type": [construct_type_str],
            "builder_name": [match_builder_name_str],
            "anken": [match_anken_str],
            "num_S": match_number,
            "ocr": {"words": words, "boxes": boxes}}


def ocr_azure_cad(image_pil: Image, computervision_client: ComputerVisionClient) -> Tuple[List, List]:
    """
    The function `ocr_azure_cad` takes an image and a ComputerVisionClient object, performs OCR using
    Azure Computer Vision, and returns the detected words along with their bounding boxes.
    
    :param image_pil: The `image_pil` parameter is of type `PIL.Image` and represents the input image in
    Pillow (PIL) format that you want to perform OCR (Optical Character Recognition) on
    :type image_pil: Image
    :param computervision_client: The `computervision_client` parameter in the `ocr_azure_cad` function
    is an instance of the Azure Computer Vision client that you will use to interact with the Computer
    Vision API provided by Azure. This client should be initialized with the necessary credentials and
    configuration to authenticate and make requests to the
    :type computervision_client: ComputerVisionClient
    :return: The function `ocr_azure_cad` returns a tuple containing two lists: 
    1. The `words` list which contains the detected text line by line.
    2. The `boxes` list which contains the bounding boxes of the detected text lines.
    """
    def cvrt_point_bb(points):
        """
        The function `cvrt_point_bb` converts a list of points to a bounding box by extracting x and y
        coordinates and finding the minimum and maximum values.
        
        :param points: It seems like the parameter `points` is a list of coordinates in the format [x1, y1,
        x2, y2, ..., xn, yn]. The function `cvrt_point_bb` takes these points and converts them into a
        bounding box format [min_x, min_y,
        :return: The function `cvrt_point_bb` returns a list containing the minimum x-coordinate, minimum
        y-coordinate, maximum x-coordinate, and maximum y-coordinate of the points provided as input.
        """
        """
        Converts a list of points to a bounding box
        """
        x_coords = [int(points[idx]) for idx in range(0, len(points), 2)]
        y_coords = [int(points[idx]) for idx in range(1, len(points), 2)]

        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    
    # Convert the image to bytes
    image_data = io.BytesIO()
    image_pil.save(image_data, format='PNG')
    image_data.seek(0)

    # Perform OCR using Azure Computer Vision
    read_response = computervision_client.read_in_stream(image=image_data, raw=True)

    # Get the operation location (URL with an ID at the end) from the response
    read_operation_location = read_response.headers["Operation-Location"]
    # Grab the ID from the URL
    operation_id = read_operation_location.split("/")[-1]

    # Call the "GET" API and wait for it to retrieve the results
    while True:
        
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break

    point_list = []
    words = []
    # Print the detected text, line by line
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                point_list.append(line.bounding_box)
                words.append(line.text)

    boxes = [cvrt_point_bb(points) for points in point_list]
    
    return words, boxes

def public_ocr_cad_apis(image_path: str,  det_draw_model: YOLO, det_code_model: YOLO) -> Tuple[dict, dict]:
    """
    This Python function `public_ocr_cad_apis` runs OCR-CAD on an image to extract basic and drawing
    information using Azure Cognitive Services.
    
    :param image_path: The `image_path` parameter in the `public_ocr_cad_apis` function is a string that
    represents the path to the image file that will be processed by the OCR-CAD APIs. This function runs
    OCR-CAD on the provided image to extract information related to drawings and codes
    :type image_path: str
    :param det_draw_model: The `det_draw_model` parameter in the `public_ocr_cad_apis` function is
    likely referring to a model used for detecting and localizing drawings in an image. This model is
    most likely based on the YOLO (You Only Look Once) architecture, which is a popular object
    :type det_draw_model: YOLO
    :param det_code_model: The `det_code_model` parameter in the `public_ocr_cad_apis` function is
    expected to be an object of the YOLO class. This model is used for detecting and extracting
    code-related information from the input image during the OCR-CAD process
    :type det_code_model: YOLO
    :return: The function `public_ocr_cad_apis` is returning a tuple containing two dictionaries:
    `basic_info_dict` and `draw_information_dict`.
    """
    """Run OCR-CAD the draw

    Args:
        engine_type (str): type of Engine
        image_path (str): path to image

    Returns:
        dict: result object
    """
    # Main process
    basic_dict, floor_dict = {}, {}
    
    endpoint = os.environ['AZURE_SHIBA_ENDPOINT_KEY']
    subscription_key = os.environ['AZURE_SHIBA_SUBSCRIPTION_KEY']
    COMPUTERVISION_CLIENT = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

    raw_image_pil = load_image(image_path)
    result_draw_df = yolo_inference(raw_image_pil, det_draw_model, type='draw')
    raw_code_df = yolo_inference(raw_image_pil, det_code_model, type='code')
    base_image_pil, draw_location_list = split_images(raw_image_pil.copy(), result_draw_df)

    # Process for basic infomation
    basic_info_dict = get_basic_info(base_image_pil, COMPUTERVISION_CLIENT)

    # Process for draw infomation
    draw_information_dict = {}
    for idx, draw_location in enumerate(draw_location_list):
        # floor_str = get_floor_index(draw_location, basic_info_dict['ocr'])
        floor_str = str(idx)
        
        code_df = raw_code_df[raw_code_df.boxes.apply(lambda b: calc_ovl(b, draw_location) > 0.5)]
        code_df.reset_index(drop=True, inplace=True)
        
        num_cell, cell_width, cell_height = get_cell_size(code_df)
        concat_image_pil = gen_grid_image(raw_image_pil, code_df, num_cell, cell_width, cell_height)
        
        draw_words, draw_boxes = ocr_azure_cad(concat_image_pil.copy(), COMPUTERVISION_CLIENT)
        extract_dict = mapping_code(raw_image_pil, draw_words, draw_boxes, code_df,
                                    num_cell, cell_width, cell_height)
        extract_dict = refine_format(extract_dict, floor_str)
        
        draw_information_dict[floor_str] = extract_dict

    basic_info_dict.pop('ocr')
    return basic_info_dict, draw_information_dict

def process_by_batch_size():
    pass

def di_azure_ocr(img_byte_arr: bytes,
                 document_analysis_client,
                 remove_check_box=True):

    # Call Azure API
    poller = document_analysis_client.begin_analyze_document("prebuilt-read", document=img_byte_arr)
    result = poller.result()
    result_json = result.to_dict()

    # document2d = ocr2structured_text(result_json)
    
    if remove_check_box:
        document = '\n'.join(line['content'].replace(':selected:', '').replace(':unselected:', '')
                             for line in result_json['pages'][0]['lines'])
    else:
        document = '\n'.join(line['content'] for line in result_json['pages'][0]['lines'])

    words = []
    poly_boxes = []
    conf_scores = []
    for line in result_json['pages'][0]['lines']:
        words.append(line['content'])
        poly_boxes.append(line['polygon'])
        line_spans = tuple((line['spans'][0]['offset'], line['spans'][0]['offset'] + line['spans'][0]['length']))
        confidence = min([word['confidence'] for word in result_json['pages'][0]['words']
                          if line_spans[0] <= word['span']['offset'] < line_spans[1]])
        conf_scores.append(confidence)
    bboxes = []
    for box in poly_boxes:
        x_coords = [point['x'] for point in box]
        y_coords = [point['y'] for point in box]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        bboxes.append([x_min, y_min, x_max, y_max])
    return  words, bboxes
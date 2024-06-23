from PIL import Image
from PIL import Image, ImageDraw

from typing import List, Tuple, Dict
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
import io
import pandas as pd
from ultralytics import YOLO
import cv2

import re
from basicsr.utils import imwrite
import glob
import numpy as np
import os
from tqdm import tqdm
import sys
sys.path.append('/mnt/d/docker_volume/room_of_nhanhuynh/ai_butler/CAD-ShibaSangyo/full_build_from_scratch_nhanhuynh/GFPGAN')
from gfpgan import GFPGANer
from src.cor_similar import *
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

import math
import regex



def load_image(image_path: str) -> Image:
    """
    This function loads an image from a specified path and converts it to a Pillow image in RGB format.
    
    Args:
        image_path (str) : The `image_path` parameter in the `load_image` function is a string that
        represents the path to the image file that you want to load as a Pillow image. This path should
        point to the location of the image file on your filesystem
    Returns:
        Image: The function returns a Pillow image object in RGB format.
    """

    image_pil = Image.open(image_path).convert('RGB')
    
    return image_pil


def cvrt_point_bb(points: List) -> List:
    """


    This Python function converts a list of points to a bounding box by extracting x and y coordinates
    and finding the minimum and maximum values for each.
    
    Args:
        points (list): It seems like you have not provided the actual list of points that you want to
        convert to a bounding box. Could you please provide the list of points so that I can assist you in
        converting them to a bounding box using the `cvrt_point_bb` function?
    
    Return:
        The function `cvrt_point_bb` is returning a list that represents a bounding box. The list
        contains the minimum x-coordinate, minimum y-coordinate, maximum x-coordinate, and maximum
        y-coordinate of the points provided as input.
    """
    x_coords = [int(x) for idx, x in enumerate(points) if idx % 2 == 0]
    y_coords = [int(y) for idx, y in enumerate(points) if idx % 2!= 0]

    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]


def ocr_azure_cad(image_pil: Image, computervision_client: ComputerVisionClient) -> Tuple[List, List]:
    def cvrt_point_bb(points):
        """
        Converts a list of points to a bounding box
        """
        x_coords = [int(points[idx]) for idx in range(0, len(points), 2)]
        y_coords = [int(points[idx]) for idx in range(1, len(points), 2)]

        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    
    # Convert the image to bytes
    image_data = io.BytesIO()
    image_pil.save(image_data, format='JPEG')
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

def load_yolo(model_path: str) -> YOLO:
    """
    This Python function loads a YOLO model from a specified path and returns it.
    
    Args:
        model_path (str): The `model_path` parameter in the `load_yolo` function is a string that
        represents the file path to the YOLO model that you want to load. This model will be used for object
        detection tasks
    
    Return: 
        An instance of the YOLO class initialized with the model located at the specified
        `model_path` is being returned.
    """

    det_model = YOLO(model_path)
    
    return det_model

def yolo_inference(image_pil: Image, model_detect: YOLO, type: str) -> pd.DataFrame:

    """
    Object detection by Yolov8
    This Python function `yolo_inference` performs object detection using YOLOv8 on a given image and
    returns the detection results in a pandas DataFrame.

    Args:
        image_pil (Image): is a Pillow image object representing the image on which
        you want to perform object detection using YOLOv8. This image will be passed to the YOLO model for
        inference

        model_detect (YOLO): is expected to be an instance of the YOLO model class. This model is used for object detection using YOLOv8 algorithm
        
        type (str): is used to specify the type of object for which you want to perform detection. It is a string parameter that can have different
        values based on the type of object you want to detect. In the provided code snippet, if the `


    Returns:
        pd.DataFrame: A pandas DataFrame containing the detection results is being returned. The DataFrame
        includes information about the detected objects such as their bounding boxes and class labels.

    """
    
    if type == 'code':
        results = model_detect.predict(source=image_pil, verbose=False)

        detect_df = pd.DataFrame({"boxes": results[0].boxes.xyxy.tolist(), "cls": results[0].boxes.cls.tolist()})
        detect_df['boxes'] = detect_df['boxes'].apply(lambda x: list(map(int, x)))
        detect_df['angle'] = detect_df['cls'].apply(lambda c: int(results[0].names[c]))
    else:


        results = model_detect.predict(source=image_pil, verbose=False, conf = 0.7)

        detect_df = pd.DataFrame({"boxes": results[0].boxes.xyxy.tolist(), "cls": results[0].boxes.cls.tolist()})
        detect_df['boxes'] = detect_df['boxes'].apply(lambda x: list(map(int, x)))
    
    return detect_df

def calc_ovl(box_small: List, box_large: List) -> float:


    """
    The function `calc_ovl` calculates the Intersection over Union (IoU) score between two bounding
    boxes.

    Args:
        box_small (list) : xmin, ymin, xmax, ymax of the smaller box
        
        box_large (list) : box_large is a list containing the coordinates of a larger bounding box. The
        coordinates represent the top-left (xmin, ymin) and bottom-right (xmax, ymax) corners of the
        bounding box
    Returns:
        The function `calc_ovl` returns the Intersection over Small Area (ios) score, which is
        calculated as the Intersection over Union (IoU) between two bounding boxes.
        float: ios (intersec on small area) score    
    """
    try:
        # Extract coordinates
        xmin1, ymin1, xmax1, ymax1 = box_small
        xmin2, ymin2, xmax2, ymax2 = box_large

        # Calculate intersection coordinates
        x_left = max(xmin1, xmin2)
        y_top = max(ymin1, ymin2)
        x_right = min(xmax1, xmax2)
        y_bottom = min(ymax1, ymax2)

        # Calculate intersection area
        intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

        # Calculate areas of each box
        box1_area = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
        box2_area = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)

        small_area = min(box1_area, box2_area)

        # Calculate IoU
        iou = intersection_area / small_area

        return iou
    except Exception as e:
        print("An error occurred while calculating IoU:", e)
        return 0.
    
def detect_circle(image_pil: Image, is_display=False) -> List[int]:
    """_summary_

    Args:
        image_pil (Image): pillow image
        is_display (bool, optional): Defaults to False.

    Returns:
        List[int]: the circle bbox
    """
    
    # Load the image
    width, _ = image_pil.size
    x_center = int(width / 2)
    draw_image = np.array(image_pil.convert('RGB'))
    image = np.array(image_pil.convert('L'))

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=100, param2=30, minRadius=10, maxRadius=50)

    # Ensure circles were found
    if circles is not None:
        # Round the circle parameters and convert to integers
        circles = np.round(circles[0, :]).astype(int)

        # Draw detected circles on the original image
        verify_x, verify_y, verify_r = 0, 0, 0
        for (x, y, r) in circles:
            verify_x, verify_y, verify_r = [x, y, r] if abs(x - x_center) < abs(verify_x - x_center) else [verify_x, verify_y, verify_r]

        circle_box = [verify_x - verify_r,
                      verify_y - verify_r,
                      verify_x + verify_r,
                      verify_y + verify_r]
        if is_display:
            # Display the image with detected circles
            cv2.circle(draw_image, (verify_x, verify_y), verify_r, (255, 0, 0), 2)
            Image.fromarray(draw_image)
        
        return circle_box
    else:
        return []



def split_images(raw_image_pil: Image, data_df: pd.DataFrame) -> Tuple[Image.Image, List]:
    """The `split_images` function takes in a raw image (as a PIL Image object) and a DataFrame containing bounding box information. It then splits the image into two parts: a basic image and a list of cropped images.
    
    Args:
        raw_image_pil: Pillow image
        data_df: DataFrame of detect result 

    
    Returns:
        basic image: pillow image exclude draws
        draw image list: list of draw images
    """
    basic_image_pil = None
    
    ymin = min(data_df.boxes.apply(lambda b: b[1]))
    ymax = max(data_df.boxes.apply(lambda b: b[3]))
    draw_height = ymax - ymin
    
    crop_xmin = 0
    crop_ymin = ymin
    crop_xmax = raw_image_pil.size[0]
    crop_ymax = ymax - int(draw_height * 0.05)
    
    purge_width, purge_height = (crop_xmax - crop_xmin, crop_ymax - crop_ymin)
    purge_image_pil = Image.new("RGB", (purge_width, purge_height), (255, 255, 255))
    basic_image_pil = raw_image_pil.copy()
    basic_image_pil.paste(purge_image_pil, (crop_xmin, crop_ymin))
    
    return basic_image_pil, data_df.boxes.tolist()

def get_cell_size(detect_df: pd.DataFrame, offset: int=7) -> Tuple[int]:
    """
    This function calculates the size of cells based on bounding box dimensions and angles in a
    DataFrame.
    
    Args:
        detect_df (pd.DataFrame): `detect_df` is a pandas DataFrame containing detection data with columns like
        'boxes' and 'angle'. The 'boxes' column contains coordinates of bounding boxes in the format [x_min,
        y_min, x_max, y_max], and the 'angle' column contains the angle of rotation for each
        
        offset (int) : The `offset` parameter in the `get_cell_size` function is used to add or subtract a
        specified value from the bounding box coordinates. In this case, the offset value is set to 7. This
        offset helps in adjusting the size of the bounding box for each detected object in the `detect,
        defaults to 7
    
    Returns:
        The function `get_cell_size` returns a tuple containing the number of cells (`num_cell`),
        the maximum width (`max_width`), and the maximum height (`max_height`) calculated based on the input
        DataFrame `detect_df` and the offset value.
    """
    max_width = 0
    max_height = 0
    offset = 10
    for det_idx, det_data in detect_df.iterrows():
        tmp_box = [int(det_data.boxes[0]) - offset,
                   int(det_data.boxes[1]) - offset,
                   int(det_data.boxes[2]) + offset,
                   int(det_data.boxes[3]) + offset]
        angle = det_data.angle
        
        box_w = tmp_box[2] - tmp_box[0] + offset
        box_h = tmp_box[3] - tmp_box[1] + offset
        
        if angle == 90:
            max_width = max(max_width, box_h)
            max_height = max(max_height, box_w)
        else:
            max_width = max(max_width, box_w)
            max_height = max(max_height, box_h)
    
    num_cell = math.ceil(math.sqrt(len(detect_df))) + 1
    
    return num_cell, max_width, max_height

def gen_grid_image(draw_image_pil: Image, detect_df: pd.DataFrame, num_cell: int, max_width: int, max_height: int, is_saved: bool = False, path_save: str = '') -> Image:
    """
    This function generates a concatenated image grid by cropping and rotating images based on detection
    data.
    
    Args:
        draw_image_pil (PIL.Image) : The `draw_image_pil` parameter is a Pillow image that will be used for
        cropping and pasting onto the concatenated image

        detect_df (pd.DataFrame) : The `detect_df` parameter is a pandas DataFrame containing detection data. It
        likely includes information about detected objects or regions in an image, such as bounding box
        coordinates and angles

        num_cell (int) : The `num_cell` parameter represents the number of cells in each row and column of
        the grid. It is used to determine how many images will be concatenated horizontally and vertically
        in the final grid image

        max_width (int) : The `max_width` parameter in the `gen_grid_image` function represents the maximum
        width of each cell in the grid when concatenating pillow images

        max_height (int) : The `max_height` parameter in the `gen_grid_image` function represents the
        maximum height of each cell in the grid when concatenating multiple images. It is used to determine
        the size of each cell in the final concatenated image grid
    
    Return:

        The function `gen_grid_image` returns a concatenated image created by pasting cropped and
        rotated images from the `draw_image_pil` based on the data in the `detect_df` DataFrame. The images
        are arranged in a grid pattern specified by the `num_cell`, `max_width`, and `max_height`
        parameters. The final concatenated image with outlined rectangles is returned.
    """

    concat_image_width = num_cell * max_width
    concat_image_height = num_cell * max_height

    concat_image = Image.new(mode='RGB', size=(concat_image_width, concat_image_height), color='white')
    offset = 5
    for row in range(num_cell):
        for column in range(num_cell):
            image_location = [(max_width * column),
                              (max_height * row)]
            
            det_idx = num_cell * row + column
            det_idx = det_idx
            if det_idx >= len(detect_df):
                continue
            angle, crop_box = detect_df.loc[det_idx, ['angle', 'boxes']].tolist()
            crop_box = [int(crop_box[0]) - offset,
                        int(crop_box[1]) - offset,
                        int(crop_box[2]) + offset +8,
                        int(crop_box[3]) + offset]
            crop_image = draw_image_pil.crop(crop_box)
            crop_image = crop_image.rotate(-angle, expand=True)
            if is_saved == True:
                crop_image.save(f'{path_save}/{int(crop_box[0])}_{int(crop_box[1])}_{int(crop_box[2])}_{int(crop_box[3])}.png')
            
            concat_image.paste(crop_image, image_location)
            ImageDraw.Draw(concat_image).rectangle([image_location[0],
                                                    image_location[1],
                                                    image_location[0] + max_width,
                                                    image_location[1] + max_height],
                                                    outline='black', width=2)
    
    return concat_image

def get_floor_index(draw_box: List, ocr_dict: Dict) -> str:
    """
    This function takes a list of bounding boxes and an OCR dictionary, extracts relevant information,
    and returns the floor index as a string.
    
    Args:

        draw_box (list) : It seems like you were about to provide some information about the `draw_box`
        parameter, but the information is missing. Could you please provide more details or an example of
        what the `draw_box` parameter represents in your code?

        ocr_dict (dict) : ocr_dict is a dictionary containing OCR (Optical Character Recognition) results,
        where the keys are column names and the values are lists of OCR results. The OCR results typically
        include information such as the detected words, bounding boxes, and other relevant data extracted
        from an image or document

    Returns:

        The function `get_floor_index` returns a string that represents the floor index. The floor
        index is extracted from the OCR data based on the provided conditions and is returned with the
        suffix "F".
    """
    ocr_df = pd.DataFrame(ocr_dict)
    ocr_df[['xmin', 'xmax', 'ymin', 'ymax']] = ocr_df['boxes'].tolist()
    
    match_df = ocr_df[ocr_df.boxes.apply(lambda b: calc_ovl(b, draw_box) >= 0.7)]
    
    digit_df = match_df[match_df.words.str.contains(r'図')]
    floor_str = regex.findall('\d*', ' '.join(digit_df.words.tolist()))[0]
    return f"{floor_str}F"



def mapping_code(raw_image_pil: Image, words: List, boxes: List, det_df: pd.DataFrame, 
                 num_cell: int, cell_width: int, cell_height: int, offset: int=10, is_display: bool=True) -> Dict:
    """
    The function `mapping_code` processes image data and OCR results to extract digit codes and
    corresponding notes for each cell in a grid.
    
    Args:

        raw_image_pil (PIL.Image) : The `raw_image_pil` parameter is expected to be an image in PIL (Python
        Imaging Library) format. This image will be used for cropping and processing in the `mapping_code`
        function

        words (list) : The `words` parameter in the `mapping_code` function is a list containing the words
        detected in the image. These words are extracted during optical character recognition (OCR)
        processing of the image

        boxes (list) : The `boxes` parameter in the `mapping_code` function is a list containing bounding
        boxes for each word detected in the OCR process. Each bounding box is represented as a list of four
        values `[xmin, xmax, ymin, ymax]`, where `xmin` and `ymin` represent the coordinates

        det_df (pd.DataFrame) : The `det_df` parameter in the `mapping_code` function seems to be a DataFrame that
        contains information about the detected objects in the image. It likely includes columns such as
        'angle' and 'boxes' which provide details about the orientation and bounding boxes of the detected
        objects

        num_cell (int) : The `num_cell` parameter in the `mapping_code` function represents the number of
        cells in both the horizontal and vertical directions that you want to divide the image into. This
        parameter is used to iterate over each cell in the image and perform certain operations within each
        cell

        cell_width (int) : The `cell_width` parameter in the `mapping_code` function represents the width of
        each cell in the grid where the image is divided. This value is used to calculate the location and
        size of the cropped images within each cell during the processing of the OCR (Optical Character
        Recognition) and object detection

        cell_height (int) : The `cell_height` parameter in the `mapping_code` function represents the height
        of each cell in the grid where the image is divided. This value is used to calculate the dimensions
        of each cell in the grid based on the provided `cell_width` and to position the cropped images
        correctly within each cell

        offset (int) : The `offset` parameter in the `mapping_code` function is used to specify the
        additional padding (in pixels) to be added or subtracted from the bounding box coordinates when
        cropping the image. This padding helps in ensuring that the cropped region includes the entire
        object of interest and some surrounding context for better, defaults to 5

        is_display (bool) : The `is_display` parameter in the `mapping_code` function is a boolean flag that
        determines whether to display the extracted code and note information during the processing of each
        cell. If set to `True`, the function will print out the code and note information for each cell as
        it processes them. If, defaults to False

    Return:
        A dictionary where the keys are the cell coordinates (as a tuple of (row, column))
        The function `mapping_code` returns a dictionary with two keys:
        1. 'digit_code': A list of integer values extracted from the code strings processed in the function.
        2. 'code': A list of strings representing notes extracted from the processed data.
    """
    
    code_ocr_df = pd.DataFrame({"words": words, "boxes": boxes})
    code_ocr_df[['xmin', 'xmax', 'ymin', 'ymax']] = code_ocr_df['boxes'].tolist()
    
    code_digit_list = []
    match_words_list = []
    cnt = 0
    for row in range(num_cell):
        for column in range(num_cell):
            image_location = [(cell_width * column),
                              (cell_height * row)]
            
            det_idx = num_cell * row + column
            if det_idx >= len(det_df):
                continue
            angle, crop_box = det_df.loc[det_idx, ['angle', 'boxes']].tolist()
            crop_box = [int(crop_box[0]) - offset,
                        int(crop_box[1]) - offset,
                        int(crop_box[2]) + offset,
                        int(crop_box[3]) + offset]
            
            crop_image = raw_image_pil.crop(crop_box)
            crop_image = crop_image.rotate(-angle, expand=True)
            
            concat_box = [image_location[0],
                          image_location[1],
                          image_location[0] + cell_width,
                          image_location[1] + cell_height]
            match_df = code_ocr_df[code_ocr_df.boxes.apply(lambda b: calc_ovl(b, concat_box) >= 0.5)]
            
            circle_box = detect_circle(crop_image)
            code_str = ''
            note_str = ''
            if circle_box:
                circle_box = [circle_box[0] + image_location[0],
                              circle_box[1] + image_location[1],
                              image_location[0] + circle_box[2],
                              image_location[1] + circle_box[3]]
                
                code_df = match_df[match_df.boxes.apply(lambda b: calc_ovl(b, circle_box) > 0.7) & (match_df.words.str.isdigit())]
                code_list = code_df.words.tolist()
                
                match_df = match_df.loc[~match_df.index.isin(code_df.index)]
                note_list = match_df[~match_df.words.str.lower().str.match(r'^[0-9\.\,\ \|]*$|^[^a-z0-9]*$')].words.tolist()
                
                # Post-processing
                code_str = code_str if not code_list else code_list[0]
                note_str = note_str if not note_list else note_list[0]
                
                code_str = regex.sub(r'[\(\)\[\]\ ]*', "", code_str)
                note_str = regex.sub(r'[\.:]*|\($', "", note_str)
                
            if code_str:
                code_digit_list.append(int(code_str[-2:]))
                match_words_list.append(note_str)
            if is_display:
                print('Code: ', code_str)
                print('Note: ', note_str)
    
    return {'digit_code': code_digit_list, 'code': match_words_list}

def refine_format(data_dict: Dict, floor_str: str) -> Dict:
    """
    This function refines the format of a given data dictionary based on a floor string.
    
    :param data_dict: A dictionary containing keys 'code' and 'digit_code'. The 'code' key holds a list
    of strings, and the 'digit_code' key holds a list of integers
    :type data_dict: Dict
    :param floor_str: Floor_str is a string that represents the floor number in a building. It is used
    as part of the key in the output dictionary to indicate the floor number along with the code
    associated with that floor
    :type floor_str: str
    :return: The function `refine_format` returns a dictionary where the keys are formatted as
    `{floor_str} {idx}` and the values are generated by joining the 'code' values from `data_dict` based
    on the condition that the corresponding 'digit_code' value is equal to `idx` for each index `idx`
    from 1 to 50.
    """
    """_summary_

    Args:
        data_dict (Dict): _description_
        floor_str (str): _description_

    Returns:
        Dict: _description_
    """
    
    parse_dict = {}
    for idx in range(1, 51):
        code_str = ' '.join([data_dict['code'][i] for i in range(len(data_dict['digit_code'])) if data_dict['digit_code'][i] == idx])
        parse_dict[f'{floor_str} {idx}'] = code_str
    
    return parse_dict

def crop_image_and_save(coordinate, image):
    """
    The function `crop_image_by_list_and_save` takes a list of coordinates and an image, crops the image
    based on the coordinates, and saves the cropped images to a specified path.
    
    :param coordinate: The `coordinate` parameter in the `crop_image_by_list_and_save` function is a
    list of lists. Each inner list contains coordinates that define a rectangular region to crop from
    the image. The coordinates are in the format `(left, upper, right, lower)` where `(left, upper)` is
    :param image: The `image` parameter in the `crop_image_by_list_and_save` function is expected to be
    an image object that you want to crop and save based on the coordinates provided in the `coordinate`
    list. The function iterates over the coordinates in the `coordinate` list, crops the image based
    """
    path_to_save = "/mnt/d/docker_volume/room_of_nhanhuynh/ai_butler/CAD-ShibaSangyo/full_build_from_scratch_nhanhuynh/tmpFile/tmp_core/"
    total_image = glob.glob('/mnt/d/docker_volume/room_of_nhanhuynh/ai_butler/CAD-ShibaSangyo/full_build_from_scratch_nhanhuynh/tmpFile/tmp_draw_restore/*')
    for i in tqdm(coordinate):
        # os.makedirs(f"{path_to_save}/{i}", exist_ok=True)
        for j in i:
            image = image.crop(tuple(j))
            image.save(f"{path_to_save}/{i}_{j}.png")

def crop_image_by_coordinate(coordinate_point, image, is_save= False, path = ""):
    """
    This function crops an image based on the provided coordinates and optionally saves the cropped
    image to a specified path.
    
    :param coordinate_point: The `coordinate_point` parameter is a tuple that represents the coordinates
    of the bounding box to crop the image. It should have 4 values: (x1, y1, x2, y2), where (x1, y1) are
    the coordinates of the top-left corner and (
    :param image: The `image` parameter in the `crop_image_by_coordinate` function is expected to be an
    image object that can be cropped using the `crop` method. This image object should support the
    `crop` method, which is commonly found in image processing libraries such as PIL (Pillow) in Python
    :param is_save: The `is_save` parameter in the `crop_image_by_coordinate` function is a boolean flag
    that determines whether the cropped image should be saved to a file or returned as a result. If
    `is_save` is set to `True`, the cropped image will be saved to a file with the specified, defaults
    to False (optional)
    :param path: The `path` parameter in the `crop_image_by_coordinate` function is a string that
    represents the directory path where the cropped image will be saved if the `is_save` parameter is
    set to `True`. If `is_save` is `False`, the function will return the cropped image without saving
    :return: If the `is_save` parameter is `False`, the function will return the cropped image.
    """
    image = image.crop(tuple(coordinate_point))
    if is_save == True:
        image.save(f"{path}/{coordinate_point[0]}_{coordinate_point[1]}_{coordinate_point[2]}_{coordinate_point[3]}.png")
    else:
        return image

def clear_cache_last_session(path_to_clear):
    """
    This function clears the cache of the last session.
    """
    if len(os.listdir(path_to_clear)) != 0:
        list_file_CAD_available = glob.glob(f'{path_to_clear}/*')
        for i in list_file_CAD_available:
            os.remove(i)
    else:
        print('Nothing cache to delete')

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return np.asarray(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR, cv2.IMREAD_COLOR)
            
def super_restore_resolution(list_image_path: str, restorer,is_saved: bool = True, des_path: str = ''):
    """
    This Python function takes a list of image paths, applies a restoration process using a provided
    restorer, and saves the restored images in a specified directory.
    
    Args:
        list_image_path (str) : The `list_image_path` parameter is the path to the directory containing the
        images that you want to restore
        
        restorer (model GFP GAN) : The `restorer` parameter in the `super_restore_resolution` function seems to be an
        object that has an `enhance` method. This method is used to enhance an input image based on certain
    
        is_saved (bool) : Saved image to diretory
        
        des_path (str) : The `des_path` parameter in the `super_restore_resolution`
    """
    if is_saved == False:
        input_img = convert_from_image_to_cv2(list_image_path)
        _, _, restored_img = restorer.enhance(
                                            input_img,
                                            has_aligned= False,
                                            only_center_face= False,
                                            paste_back= True,
                                            weight= 0.1)
        return restored_img

    img_list = sorted(glob.glob(f"{list_image_path}/*"))
    os.makedirs(f"{des_path}", exist_ok=True)
    clear_cache_last_session(f"{des_path}")
    for img_path in tqdm(img_list):
        img_name = os.path.basename(img_path)
        # print(f'Processing {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # restore all component if necessary
        _, _, restored_img = restorer.enhance(
            input_img,
            has_aligned= False,
            only_center_face= False,
            paste_back= True,
            weight= 0.5)
        
        if restored_img is not None:
            if ext == 'auto':
                extension = 'png'
            else:
                extension = ext
                save_restore_path = os.path.join(f'{des_path}', f"{basename}{extension}")
                imwrite(restored_img, save_restore_path)
    print("Done")
    return f"{list_image_path}_restore/"

def reformat_code(result: str = ''):

    if result.startswith('K') or result.startswith('M') or result.startswith('SH'):
        result = '\t\t\t\t\t上\n' + result +' \n\t\t\t\t\t下'
    else:
        result = result
    return result

def convert_parentheses_string_to_int(s: str = ''):
    # Remove the parentheses
    s = s.strip('()')
    # Convert the remaining string to an integer
    return int(s)


# def fill_missing_row(list_original:list = []):
#     new_list_appending = []
#     for i in list_original:
#         component = i.split('|')
#         stt = convert_parentheses_string_to_int(s=component[0])
#         if len(new_list_appending) == 0:
#             new_list_appending.append(stt+'|'+component[1])
#         else:
#             for j in new_list_appending:

def fill_missing_elements(data):
    check_duplicate_list = []
    # Extract the numbers from the data
    
    numbers = [int(item.split(')')[0][1:]) for item in data]
    if numbers not in check_duplicate_list:
        # Determine the complete range of numbers
        full_range = range(min(numbers), max(numbers) + 1)
        
        # Create a dictionary from the original data for quick lookup
        data_dict = {int(item.split(')')[0][1:]): item for item in data}
        
        # Construct the complete list with missing elements filled with "|None"
        result = []
        for num in full_range:
            if num not in check_duplicate_list:
                if num in data_dict:
                    result.append(data_dict[num])
                else:
                    result.append(f"({num})|None")
                check_duplicate_list.append(num)
        
        return result

def process_code_cad_1(str_code: str = '', list_sample: list = []):
    if str_code in list_sample:
        result = reformat_code(str_code)
        return result
    elif str_code not in list_sample:
        list_score_similarity = []
        print('The system not found this code this dict so we will calculate similarity of it with all code in dict')
        for idx_core_cad in list_sample:
            score_similarity = calculator_similarity_str(str_1= str_code, str_2= idx_core_cad)
            list_score_similarity.append(score_similarity)
        idx_code_max = list_score_similarity.index(max(list_score_similarity))
        print(idx_code_max)
        result = list_sample[idx_code_max]
        print(result)
        result = reformat_code(result= result)
        print(result)
        return result


def process_code_cad_2(str_code: str = '', list_sample: list = []):
    if str_code in list_sample:
        result = reformat_code(str_code)
    elif str_code not in list_sample:
        list_score_similarity = []
        print('The system not found this code this dict so we will calculate similarity of it with all code in dict')
        for idx_core_cad in list_sample:
            list_score_similarity = benmarking_metric_similar(str_1= str_code, str_2= idx_core_cad, list_conf= list_score_similarity)
        df_metric = pd.DataFrame(list_score_similarity, columns=['levenshtein', 'lcs', 'ngram', 'cosine'])
        id_score_best = select_feature_from_dataframe(df_metric= df_metric)
        result = list_sample[id_score_best]
        result = reformat_code(result= result)
        return result


def reconfig_list(str_process: list = []) -> Tuple[int,str]:
    """
    The function `reconfig_list` takes a list of strings as input, converts the first string to an
    integer using a helper function, and reformats the second string before returning a tuple of the
    integer and the reformatted string.
    
    Args:

        str_process (list) : The `str_process` parameter is a list that contains two elements. The first
        element is a string that represents a parentheses string that needs to be converted to an integer.
        The second element is a string that represents code that needs to be reformatted. The function
        `reconfig_list` takes this list
    Return:

        A tuple containing an integer and a string is being returned.
    """
    idx_int = convert_parentheses_string_to_int(str_process[0])
    code = reformat_code(str_process[1])
    return idx_int, code


def upscale_image(image_path: str, restorer,is_saved: bool = True, des_path: str = ''):


    img_name = os.path.basename(image_path)
    # print(f'Processing {img_name} ...')
    basename, ext = os.path.splitext(img_name)
    input_img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # restore all component if necessary
    _, _, restored_img = restorer.enhance(
        input_img,
        has_aligned= False,
        only_center_face= False,
        paste_back= True,
        weight= 0.5)
    
    if restored_img is not None:
        if ext == 'auto':
            extension = 'jpg'
        else:
            extension = ext
        save_restore_path = os.path.join(f'{des_path}', f"{basename}{extension}")
        imwrite(restored_img, save_restore_path)
    print("Done")
    return save_restore_path

def combine_rephrase_dict(dict_result: Dict = {}) -> Dict:
    """
    (NON FILL MISSING !!!!!!!!)
    This function takes a dictionary with 'digit_code' and 'code' keys, pairs up the elements of the two
    lists, sorts the pairs based on the first element, and then refines the output by creating a new
    dictionary with the sorted pairs.
    
    Args:
        dict_result (Dict) : It looks like you have a function `combine_rephrase_dict` that takes a
        dictionary `dict_result` as input. The function pairs up the elements of two lists within the
        dictionary, sorts the pairs based on the first element, and then separates them back into two lists.
        Finally, it refines
    
    Return:
        The function `combine_rephrase_dict` is returning the original input dictionary
        `dict_result`.
    """
    list_ids = dict_result['digit_code']
    list_code = dict_result['code']

    # Pair up the elements of the two lists
    paired_list = list(zip(list_ids,list_code))

    # Sort the pairs based on the first element
    paired_list.sort(key=lambda x: x[0])

    # Separate the sorted pairs back into two lists
    list_ids, list_code = zip(*paired_list)
    dict_refine= {}
    # Convert the lists back to lists (since zip returns tuples)
    list_ids = list(list_ids)
    list_code = list(list_code)
    for i in tqdm(range(len(list_code)), desc= "Refining ouput"):
        dict_refine[f"{list_ids[i]}"] = list_code[i]
    return dict_result

def combine_rephrase_dict_2(dict_result: Dict = {}) -> Dict:
    """
    (HAVE FILL MISSING !!!!!!!)
    The function `combine_rephrase_dict_2` takes a dictionary with 'digit_code' and 'code' lists, pairs
    and sorts the elements based on 'digit_code', and creates a refined dictionary with consecutive keys
    from the minimum to maximum 'digit_code' values.
    
    Args:
        dict_result (Dict) : The function `combine_rephrase_dict_2` takes a dictionary `dict_result` as
        input, which should contain two keys: 'digit_code' and 'code'. The value corresponding to
        'digit_code' should be a list of digit codes, and the value corresponding to 'code' should be
        a list of corresponding codes.
    
    Return:
        The function `combine_rephrase_dict_2` returns a dictionary `dict_refine` that contains
        refined output based on the input dictionary `dict_result`. The dictionary `dict_refine` is created
        by pairing up elements from the 'digit_code' and 'code' lists in the input dictionary, sorting them
        based on the 'digit_code', and then creating a new dictionary where keys are sequential integers
    """
    list_ids = dict_result['digit_code']
    list_code = dict_result['code']

    # Pair up the elements of the two lists
    paired_list = list(zip(list_ids, list_code))

    # Sort the pairs based on the first element
    paired_list.sort(key=lambda x: x[0])

    list_ids, list_code = zip(*paired_list)

    # Separate the sorted pairs back into two lists
    list_ids, list_code = zip(*paired_list)

    dict_refine = {}
    min_id = min(list_ids)
    max_id = max(list_ids)

    for i in tqdm(range(min_id, max_id + 1), desc="Refining output"):
        if i in list_ids:
            dict_refine[i] = list_code[list_ids.index(i)]
        else:
            dict_refine[i] = ""

    return dict_refine
# def save_to_path(file):
    
def preprocessing_str(basic_info_dict: Dict = {}) -> Dict:
    if 'Ⅶ' in basic_info_dict['construct_type'][0] or 'VII' in basic_info_dict['construct_type'][0]:
        if "Ⅶ" in basic_info_dict['construct_type'][0]:
            basic_info_dict['construct_type'][0] = basic_info_dict['construct_type'][0].replace('Ⅶ', 'VII')
        basic_info_dict['construct_type'] = basic_info_dict['construct_type'][0].replace('VII','7')
    elif 'VI' in basic_info_dict['construct_type'][0]:
        basic_info_dict['construct_type'] = basic_info_dict['construct_type'][0].replace('VI', '6')
    return basic_info_dict


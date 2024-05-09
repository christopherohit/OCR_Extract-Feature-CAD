from PIL import Image
from typing import List, Tuple, Dict
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
import io
import pandas as pd
from ultralytics import YOLO
import cv2


def load_image(image_path: str) -> Image:
    """
    This function loads an image from a specified path and converts it to a Pillow image in RGB format.
    
    :param image_path: The `image_path` parameter in the `load_image` function is a string that
    represents the path to the image file that you want to load as a Pillow image. This path should
    point to the location of the image file on your filesystem
    :type image_path: str
    :return: a Pillow image object.
    """
    """Load image path to pillow image

    Args:
        image_path (str):  path to image

    Returns:
        Image: pillow image
    """
    image_pil = Image.open(image_path).convert('RGB')
    
    return image_pil

def cvrt_point_bb(points: List) -> List:
    """
    Converts a list of points to a bounding box
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
    det_model = YOLO(model_path)
    
    return det_model

def yolo_inference(image_pil: Image, model_detect: YOLO, type: str) -> pd.DataFrame:
    """Object detection by Yolov8

    Args:
        image_pil (Image): pillow image
        model_detect (YOLO): model detect
        type (str): type of object

    Returns:
        pd.DataFrame: detect result
    """
    results = model_detect.predict(source=image_pil, verbose=False)
    if type == 'code':
        detect_df = pd.DataFrame({"boxes": results[0].boxes.xyxy.tolist(), "cls": results[0].boxes.cls.tolist()})
        detect_df['boxes'] = detect_df['boxes'].apply(lambda x: list(map(int, x)))
        detect_df['angle'] = detect_df['cls'].apply(lambda c: int(results[0].names[c]))
    else:
        detect_df = pd.DataFrame({"boxes": results[0].boxes.xyxy.tolist(), "cls": results[0].boxes.cls.tolist()})
        detect_df['boxes'] = detect_df['boxes'].apply(lambda x: list(map(int, x)))
    
    return detect_df

def calc_ovl(box_small: List, box_large: List) -> float:
    """Calculate 

    Args:
        box_small (List): xmin, ymin, xmax, ymax
        box_large (List): xmin, ymin, xmax, ymax

    Returns:
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

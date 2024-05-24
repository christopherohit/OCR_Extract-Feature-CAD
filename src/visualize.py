from PIL import Image, ImageDraw





def draw_bbox_on_img(coordinate_point, image_pil):
    draw = ImageDraw.Draw(image_pil)
    draw.rectangle(((coordinate_point[0], coordinate_point[1]), (coordinate_point[2], coordinate_point[3])), fill="black")
    return image_pil

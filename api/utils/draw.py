from PIL import Image, ImageDraw
import io
import os
import base64
import cv2
from utils.lib import get_unique_id

def draw_image_detections(prediction_result, image_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for bounding_box in prediction_result:
        x0 = bounding_box['x'] - bounding_box['width'] / 2
        x1 = bounding_box['x'] + bounding_box['width'] / 2
        y0 = bounding_box['y'] - bounding_box['height'] / 2
        y1 = bounding_box['y'] + bounding_box['height'] / 2
        box = (x0, y0, x1, y1)
        draw.rectangle(box, outline="blue", width=1)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return image_base64

def draw_video_detections(video_results, video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    _, extension = os.path.splitext(video_file_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    predicted_video_path = "/api/predictions/videos/" + get_unique_id() + extension
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(predicted_video_path, fourcc, fps, (frame_width, frame_height))

    total_detections_per_frame = {}

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number in video_results['frame_offset']:
            index = video_results['frame_offset'].index(frame_number)
            total_detections_per_frame[frame_number] = len(video_results['cells-detections'][index]['predictions'])

            for prediction in video_results['cells-detections'][index]['predictions']:
                x_center, y_center = prediction['x'], prediction['y']
                width, height = prediction['width'], prediction['height']
                x0, y0 = int(x_center - width / 2), int(y_center - height / 2)
                x1, y1 = int(x_center + width / 2), int(y_center + height / 2)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 1)

        if frame_number in total_detections_per_frame:
            text = f"Total Detections: {total_detections_per_frame[frame_number]}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_color = (255, 255, 255)  # White
            font_thickness = 2

            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_width, text_height = text_size

            rectangle_bgr = (0, 0, 0)  # Black

            rect_x0 = 10
            rect_y0 = frame_height - text_height - 10
            rect_x1 = rect_x0 + text_width + 10
            rect_y1 = frame_height - 5

            cv2.rectangle(frame, (rect_x0, rect_y0), (rect_x1, rect_y1), rectangle_bgr, cv2.FILLED)
            cv2.putText(frame, text, (rect_x0, rect_y1 - 5), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return predicted_video_path

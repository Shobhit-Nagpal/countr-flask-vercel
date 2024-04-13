from flask import Flask, request, jsonify
import tempfile
from markupsafe import escape
from utils.model import load_model
from utils.draw import draw_image_detections, draw_video_detections
from utils.lib import get_mime_type, get_unique_id, convert_to_h264
from PIL import Image
import io
import base64
import os
from supabase import create_client, Client
import magic
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:3000", "https://liviteq-demo.vercel.app"])
model = load_model()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

@app.get("/")
def index():
    return "Server is up and running!"


@app.post("/count/image")
def count_image():
    tmp_file_path = None
    try:
        if 'media' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['media']

        if file.filename == '':
            return jsonify({"error": "No file selected for uploading"}), 400
        
        image_data = file.read()  # Read the file in binary mode
        image_base64 = base64.b64encode(image_data).decode('utf-8')  # Encode as base64 string
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(image_data)
            tmp_file_path = tmp_file.name

        prediction_result = model.predict(
            tmp_file_path, confidence=40, overlap=30)

        prediction_image = draw_image_detections(prediction_result, tmp_file_path)

        os.remove(tmp_file_path)

        return jsonify({
            "type": "image",
            "src": prediction_image,
            "totalCount": len(prediction_result)
        }), 200
    except Exception as e:
        if 'tmp_file_path' in locals() and tmp_file_path:
            os.remove(tmp_file_path)
        error_message = str(e)
        return jsonify({"error": error_message}), 500


@app.post("/count/video")
def count_video():
    try:
        if 'media' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        video = request.files['media']

        if video.filename == '':
            return jsonify({"error": "No file selected for uploading"}), 400

        unique_id = get_unique_id()
        tmp_file_path = os.path.join('/tmp', unique_id + ".mp4")
        video.save(tmp_file_path)

        cap = cv2.VideoCapture(tmp_file_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        job_id, signed_url, expire_time = model.predict_video(
            tmp_file_path,
            fps=video_fps,
            prediction_type="batch-video",
        )

        results = model.poll_until_video_results(job_id)

        predicted_video_path = draw_video_detections(results, tmp_file_path)
        final_video_path = convert_to_h264(predicted_video_path)
        predicted_video_filename = os.path.basename(final_video_path)
        
        mime_type = get_mime_type(final_video_path)

        public_path = f"public/{predicted_video_filename}"

        with open(final_video_path, 'rb') as f:
            supabase.storage.from_("videos").upload(file=f,path=public_path, file_options={"content-type": mime_type})


        res = supabase.storage.from_('videos').get_public_url(public_path)
        predicted_video_public_url = res

        os.remove(tmp_file_path)
        os.remove(final_video_path)

        return jsonify({"type": "video", "src": predicted_video_public_url}), 200
    except Exception as e:
        if 'tmp_file_path' in locals() and tmp_file_path:
            os.remove(tmp_file_path)
        if 'final_video_path' in locals() and final_video_path:
            os.remove(final_video_path)
        error_message = str(e)
        return jsonify({"error": error_message}), 500


if __name__ == '__main__':
    app.run(debug=True)

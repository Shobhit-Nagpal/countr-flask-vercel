import string
import random
import magic
import subprocess
import os

def get_unique_id():
    characters = string.ascii_letters + string.digits
    unique_id = ''.join(random.choice(characters) for _ in range(8))
    return unique_id

def get_mime_type(file_path):
    mime = magic.Magic(mime=True)
    mimetype = mime.from_file(file_path)
    return mimetype



def convert_to_h264(input_video_path):
    output_video_path = input_video_path.replace('.mp4', '_h264.mp4')
    command = [
        'ffmpeg',
        '-i', input_video_path,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        output_video_path
    ]
    subprocess.run(command, check=True)
    os.remove(input_video_path)
    return output_video_path

from dotenv import load_dotenv
from roboflow import Roboflow
import os

load_dotenv()

def load_model():
    rf = Roboflow(api_key=os.getenv("MODEL_API_KEY"))
    project = rf.workspace().project(os.getenv("PROJECT_NAME"))
    model = project.version(os.getenv("MODEL_VERSION")).model

    return model

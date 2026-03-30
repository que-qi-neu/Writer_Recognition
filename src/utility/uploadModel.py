from huggingface_hub import login, upload_folder
from config import settings

# (optional) Login with your Hugging Face credentials
login()

# Push your model files
upload_folder(folder_path=settings.MODEL_DIR / 'upload', repo_id="queModels/q_writer_style_recognition", repo_type="model")

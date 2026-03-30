from huggingface_hub import hf_hub_download, snapshot_download
from config import settings

def downloadModel(repo_ID:str, modelFileName = ''):
    print("Downloading model...")

    path = ''

    if modelFileName  == '':
        path = snapshot_download(
        repo_id=repo_ID, 
        local_dir=settings.MODEL_DIR,                
        local_dir_use_symlinks=False             
        )
    else:
        path = hf_hub_download(
        repo_id=repo_ID, 
        filename=modelFileName, 
        local_dir=settings.MODEL_DIR,
        local_dir_use_symlinks=False
    )

    print(f"Success! Model downloaded to: {path}")
    return path

def downloadWriterModelsQ():
    return downloadModel('queModels/q_writer_style_recognition')

def downloadSingleModel(modelFileName):
    return downloadModel('queModels/q_writer_style_recognition', modelFileName)
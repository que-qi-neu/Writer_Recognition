import os
from config import settings
from pathlib import Path

def getFolderFiles(folderPath):
    return [os.path.join(folderPath, f) for f in os.listdir(folderPath)]

def getCVL_label(fullPathStr):
    filename = os.path.basename(fullPathStr)             
    writer = filename.split('-')[0]
    return writer

def getCVL_PathLabels(directory):
    paths = getFolderFiles(directory)
    labels = [ getCVL_label(file) for file in paths]
    return paths, labels

def getDir_folders(directory):
    folders = [str(folder) for folder in directory.iterdir() if folder.is_dir()]
    return folders

def getFolder_labels(folderPaths):
    return [Path(path).name for path in folderPaths]

# print(getFolder_labels(getDir_folders(settings.ROOT_DIR)))


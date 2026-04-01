from src.utility import downloadModel
from src.training.TripletEmbedTrain import TripletEmbeddModel
from src.ui.DataAccess import EmbedModelDataAccess
from config import settings
import torch
import torch.nn.functional as F

modelPath = settings.MODEL_DIR / 'TripletEmbed.ckpt'
model = None
if modelPath.exists():
    model = TripletEmbeddModel.load_from_checkpoint(modelPath)
else:
    tripletModel = downloadModel.downloadSingleModel('TripletEmbed.ckpt')
    model = TripletEmbeddModel.load_from_checkpoint(tripletModel)

writer1_1 = model.get_embedding(settings.DATA_DIR / 'custom/0950-2.tif')
writer1_2 = model.get_embedding(settings.DATA_DIR / 'custom/0950-3.tif')
writer2 = model.get_embedding(settings.DATA_DIR / 'custom/0952-1.tif')

# print(writer1_1, writer1_2, writer2, sep='\n')

print('Vector Distance for images among Same writer: ', F.pairwise_distance(writer1_1, writer1_2))
print('Vector Distance for images among Different writer: ', F.pairwise_distance(writer1_1, writer2))

EmbedDB = EmbedModelDataAccess(model)
# EmbedDB.UpsertFolder(settings.DATA_DIR / "CVL-cropped/train")
# EmbedDB.UpsertFolder(settings.DATA_DIR / "CVL-cropped/val")
results = EmbedDB.NearestImages(settings.DATA_DIR / 'custom/0950-2.tif')
print(results['ids'], results['distances'])
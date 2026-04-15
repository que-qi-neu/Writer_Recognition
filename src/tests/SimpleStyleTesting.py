from src.utility import downloadModel
from src.training.TripletEmbedTrain import TripletEmbeddModel
from src.ui.DataAccess import EmbedModelDataAccess
from config import settings
from src.training import ImageLabels
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
writer2 = model.get_embedding(settings.DATA_DIR / 'custom/0001-2.tif')

# print(writer1_1, writer1_2, writer2, sep='\n')

print('Vector Distance for images among Same writer: ', F.pairwise_distance(writer1_1, writer1_2))
print('Vector Distance for images among Different writer: ', F.pairwise_distance(writer1_1, writer2))

EmbedDB = EmbedModelDataAccess(model)
# EmbedDB.vectorDB.client.reset()
# EmbedDB.UpsertFolder(settings.DATA_DIR / "custom")
# EmbedDB.UpsertFolder(settings.DATA_DIR / "CVL-cropped/stored")
# EmbedDB.UpsertFolder(settings.DATA_DIR / "CVL-cropped/train")
# EmbedDB.UpsertFolder(settings.DATA_DIR / "CVL-cropped/val")
results = EmbedDB.NearestImages(settings.DATA_DIR / 'custom/0950-2.tif', k=8)
print(results['ids'], results['distances'])

# Uncomment this after downloading CVL database. The CVL database images are not uploaded to github due to
# its size
# Also the folder name must match exactly like these, the /stored represent images stored in vector db, 
# unstored means images from same writers but not stored in db, and val uses complete different set of writers
imagesSame = ImageLabels.getFolderFiles(settings.DATA_DIR / "CVL-cropped/unstored")
imagesDiff = ImageLabels.getFolderFiles(settings.DATA_DIR / "CVL-cropped/val")
imgSameResult = []
imgDiffResult = []
accuracySame = 0
accuracyDiff = 0
for img in imagesSame:
    imgSameResult.append(EmbedDB.averageDistances(imageIn=img, k=1))
    if EmbedDB.existingWriter(img):
        accuracySame += 1
for img in imagesDiff[:250]:
    imgDiffResult.append(EmbedDB.averageDistances(imageIn=img, k=1))
    if not EmbedDB.existingWriter(img):
        accuracyDiff += 1

print(f'average distance for {len(imagesSame)} images of existing writer', sum(imgSameResult)/len(imagesSame))
print(f'Among {len(imagesSame)} images of existing writer, {accuracySame} were predicted as existing')
print(f'average distance for {len(imagesDiff)} images of unknown writer', sum(imgDiffResult)/len(imgDiffResult))
print(f'Among {len(imagesDiff)} images of unknown writer, {accuracyDiff} were predicted as Non Existing')

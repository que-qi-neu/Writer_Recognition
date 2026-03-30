from src.utility import downloadModel
from src.training.TripletEmbedTrain import TripletEmbeddModel
from config import settings
import torch
import torch.nn.functional as F

tripletModel = downloadModel.downloadSingleModel('TripletEmbed.ckpt')
model = TripletEmbeddModel.load_from_checkpoint(tripletModel)

writer1_1 = model.get_embedding(settings.DATA_DIR / 'custom/0950-2.tif')
writer1_2 = model.get_embedding(settings.DATA_DIR / 'custom/0950-3.tif')
writer2 = model.get_embedding(settings.DATA_DIR / 'custom/0952-1.tif')

# print(writer1_1, writer1_2, writer2, sep='\n')

print('Vector Distance for images among Same writer: ', F.pairwise_distance(writer1_1, writer1_2))
print('Vector Distance for images among Different writer: ', F.pairwise_distance(writer1_1, writer2))
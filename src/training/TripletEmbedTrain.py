import torch
import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, ViTForImageClassification, TrainingArguments, Trainer, PreTrainedModel, PretrainedConfig, AutoConfig
from PIL import Image
from config import settings
from pytorch_metric_learning import losses, miners, samplers
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as py_light
from lightning.pytorch.loggers import CSVLogger 
from torchvision import transforms
from PIL import ImageOps

import requests

from src.training.ImageLabels import *
from src.training.EmbeddingModel import EmbeddingModel
from src.utility import utils

cvl_path = settings.DATA_DIR / "CVL-cropped/train"
cvl_val_path = settings.DATA_DIR / "CVL-cropped/val"
cvl_stats_path = settings.DATA_DIR / "CVL-cropped/stats"

pretrained_modelName = 'microsoft/trocr-large-handwritten'

class ImagePadding:
    def __call__(self, img):
        dimension = max(img.size)
        return ImageOps.pad(img, (dimension, dimension), color=(255, 255, 255))

def img_pixel_Transformer():
    transformer = transforms.Compose([
            #padding the image into square one
            ImagePadding(),
            transforms.Resize((384, 384)), 
            transforms.ToTensor(),        
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
        ])
    return transformer

def img_augmented_transforms():
    return transforms.Compose([
        ImagePadding(),
        transforms.RandomResizedCrop((384, 384), scale=(0.8, 1.0), ratio=(1, 1)),         
        transforms.RandomRotation(degrees=(-10, 10)),       
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


class TripletEmbeddModel(py_light.LightningModule, EmbeddingModel):
    def __init__(self, embedding_length=512, lr=5e-5):
        super().__init__()
        self.save_hyperparameters()
        self.imgTansformer = img_pixel_Transformer()
        
        base_model = VisionEncoderDecoderModel.from_pretrained(pretrained_modelName)
        self.encoder = base_model.encoder
        
        self.encoder.gradient_checkpointing_enable()

        # defines sequence of execution for last few layers during training
        # since this is only the head of network, it contains only a few layer to interpretate encoding output
        self.projection = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_length)
        )
        
        self.lossFunc = losses.TripletMarginLoss(margin=0.3)
        self.miner = miners.MultiSimilarityMiner()

    def forward(self, pixel_values):
        # outputs.last_hidden_state shape: [batch_size, sequence_length, embedding_vectorLength]
        # sequence length is length of grids. Vision encoder breaks input pixel into hundreds of small grids, and adds CLS token at gridList[0] 
        styleEmbedding = self.encoder(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        embeddings = self.projection(styleEmbedding)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def training_step(self, batch, batch_idx):
        loss = self.getLoss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.getLoss(batch)
        self.log("validation_loss", loss, prog_bar=True)
        return loss
    
    def getLoss(self, batch):
        x, y = batch["pixel_values"], batch["labels"]
        embeddings = self(x)
        
        loss = self.lossFunc(embeddings, y, self.miner(embeddings, y))

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
    
    def get_Img_Transformer(self, rgb_Image):
        return self.imgTansformer
    
    def get_embedding(self, imageIn):
        
        image = utils.getRGBImage(imageIn)

        pixelValues = self.imgTansformer(image)       
        pixelValues = pixelValues.unsqueeze(0)       
        pixelValues = pixelValues.to(self.device)

        self.eval()
        with torch.no_grad():
            embedding = self(pixelValues) 
            
        return embedding
    
    def get_list_embeddings(self, image_list, batch_size=32):
        self.eval() 
        embeddings = []
        for i in range(0, len(image_list), batch_size):
            pixelValuesList = []
            batch_images = image_list[i : i + batch_size]
            for img in batch_images:
                image = utils.getRGBImage(img)
                tensor = self.imgTansformer(image)
                pixelValuesList.append(tensor)
                
            batchTensor = torch.stack(pixelValuesList)           
            batchTensor = batchTensor.to(self.device)
            
            with torch.no_grad():
                embeddingsBatch = self(batchTensor)
                embeddings.extend(embeddingsBatch.detach().cpu().tolist())

            torch.cuda.empty_cache()
            
        return embeddings


    
        
def map_label(filePaths, processor):  
    images = [Image.open(path).convert("RGB") for path in filePaths]
    pixelValues = processor(images, return_tensors="pt", padding=True).pixel_values
    pixelValuesList = [e for e in pixelValues]

    return {"pixel_values": pixelValuesList}

class StyleDataset(Dataset):
    def __init__(self, file_paths, labels, pixel_values=None, transformer = img_pixel_Transformer):
        self.file_paths = file_paths
        self.labels = labels
        self.pixel_values = pixel_values
        self.pixel_transform = transformer()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_pixel_values = None
        if self.pixel_values is None:
            img_path = self.file_paths[idx]
            image = Image.open(img_path).convert("RGB")        
            image_pixel_values = self.pixel_transform(image)
        else:
            image_pixel_values = self.pixel_values[idx]
        
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        
        return {"pixel_values": image_pixel_values, "labels": label}
    
def make_DataLoader(directory, labelFunction, batch=32, transformer = img_pixel_Transformer):
    # hugging face built in processor
    processor = TrOCRProcessor.from_pretrained(pretrained_modelName)

    Paths, Labels = labelFunction(directory)
    DS = StyleDataset(Paths, Labels, transformer=transformer)

    # m represent how many images chosen from the anchor/N/P
    sampler = samplers.MPerClassSampler(
    labels=Labels, 
    m=4, 
    batch_size=batch,
    length_before_new_iter=len(DS)  #without this, sampler will try to anchor all labels, causing large steps each epoch. This tell how many step each epoch
    )

    dataLoader = DataLoader(
        DS, 
        batch_size=batch, 
        sampler=sampler,
        pin_memory=True,
        num_workers=8,
        persistent_workers=True
    )

    return dataLoader


#In windows, when we use multi CPU it's going to start all over again in child process 
# causing infinite loop. Must specify this to prevent re-runing
if __name__ == '__main__':

    model = TripletEmbeddModel()

    batch = 16
    csv_logger = CSVLogger(save_dir=cvl_stats_path, name="training_log")

    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1, 
        precision="bf16-mixed", 
        max_epochs=20,
        logger=csv_logger,
        log_every_n_steps=10,
        accumulate_grad_batches = 2
    )

    trainLoader = make_DataLoader(cvl_path, getCVL_PathLabels, batch, transformer=img_augmented_transforms)
    valLoader = make_DataLoader(cvl_val_path, getCVL_PathLabels, batch)
    
    
    trainer.fit(model, train_dataloaders=trainLoader, val_dataloaders=valLoader)


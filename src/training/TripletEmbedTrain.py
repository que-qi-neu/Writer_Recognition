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

import requests

from src.training.ImageLabels import *
from src.training.EmbeddingModel import EmbeddingModel

cvl_path = settings.DATA_DIR / "CVL-cropped/train"
cvl_val_path = settings.DATA_DIR / "CVL-cropped/val"
cvl_stats_path = settings.DATA_DIR / "CVL-cropped/stats"

pretrained_modelName = 'microsoft/trocr-large-handwritten'


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
        
        self.loss_fct = losses.TripletMarginLoss(margin=0.3)
        self.miner = miners.MultiSimilarityMiner()

    def forward(self, pixel_values):
        # outputs.last_hidden_state shape: [batch_size, sequence_length, embedding_vectorLength = 1024]
        # batch means how many inputs to train every time
        # sequence length is length of grids. Vision encoder breaks input pixel into hundreds of small grids, and adds CLS token at gridList[0] 
        # embedding vector length is 1024 
        style_embedding = self.encoder(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        embeddings = self.projection(style_embedding)
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
        
        # 1. Mine the triplets from the batch labels
        indices_tuple = self.miner(embeddings, y)
        # 2. Calculate Triplet Loss
        loss = self.loss_fct(embeddings, y, indices_tuple)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
    
    def get_Img_Transformer(self, rgb_Image):
        return self.imgTansformer
    
    def get_embedding(self, imageIn):
        
        if isinstance(imageIn, (str, Path)):
            image = Image.open(imageIn).convert("RGB")
        else:
            image = imageIn.convert("RGB")

        pixel_values = self.imgTansformer(image)       
        pixel_values = pixel_values.unsqueeze(0)       
        pixel_values = pixel_values.to(self.device)

        self.eval()
        with torch.no_grad():
            embedding = self(pixel_values) 
            
        return embedding


    

def img_pixel_Transformer():
    transformer = transforms.Compose([
            transforms.Resize((384, 384)), # TrOCR-Large requires exactly 384x384
            transforms.ToTensor(),         # Converts to PyTorch Tensor & scales to 0-1
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # TrOCR defaults
        ])
    return transformer

        
def map_label(filePaths, processor):  
    images = [Image.open(path).convert("RGB") for path in filePaths]
    pixel_values = processor(images, return_tensors="pt", padding=True).pixel_values
    pixel_values_List = [e for e in pixel_values]

    return {"pixel_values": pixel_values_List}

class StyleDataset(Dataset):
    def __init__(self, file_paths, labels, pixel_values=None):
        self.file_paths = file_paths
        self.labels = labels
        self.pixel_values = pixel_values
        self.pixel_transform = img_pixel_Transformer()

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
    
def make_DataLoader(directory, labelFunction, batch=32):
    # hugging face processor to change image pixel, use this for CPU image processing
    processor = TrOCRProcessor.from_pretrained(pretrained_modelName)

    Paths, Labels = labelFunction(directory)
    DS = StyleDataset(Paths, Labels)

    # m represent how many images chosen from the anchor/N/P
    sampler = samplers.MPerClassSampler(
    labels=Labels, 
    m=4, 
    batch_size=batch,
    length_before_new_iter=len(DS) * 2 #without this, sampler will try to anchor all labels, causing large steps each epoch. This tell how many step each epoch
    )

    dataLoader = DataLoader(
        DS, 
        batch_size=batch, 
        sampler=sampler, #make sure triplet is correct format
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
        max_epochs=5,
        logger=csv_logger,
        log_every_n_steps=10,
        accumulate_grad_batches = 2
    )

    trainLoader = make_DataLoader(cvl_path, getCVL_PathLabels, batch)
    valLoader = make_DataLoader(cvl_val_path, getCVL_PathLabels, batch)
    
    
    trainer.fit(model, train_dataloaders=trainLoader, val_dataloaders=valLoader)


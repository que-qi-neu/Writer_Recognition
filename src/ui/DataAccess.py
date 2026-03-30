from src.training.ImageLabels import *
from src.training.EmbeddingModel import EmbeddingModel
from src.training.TripletEmbedTrain import TripletEmbeddModel
from pathlib import Path
from PIL import Image
import numpy as np
import faiss
import torch

class EmbedModelAccess():
    def __init__(self, model : EmbeddingModel):
        self.model = model
        self.dbIndex = faiss.IndexFlatL2(self.model.get_Dimension())

    def get_Embedding_Vector(self, path:str):
        return self.model.get_embedding(Image.open(path))
    
    def get_Embedding_Vector(self, paths:list[str]):
        return [self.model.get_Embedding(e) for e in paths]

    def store_Embeddings(self, paths:list[str]):
        embeddings = self.get_Embedding_Vector(paths)
        self.dbIndex.add(embeddings)

    def getNearest_Vectors(self, vector:list[float], k=5):
        distances, indices = self.dbIndex.search(vector, k)

        


    


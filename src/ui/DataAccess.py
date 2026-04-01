from src.training import ImageLabels
from src.training.EmbeddingModel import EmbeddingModel
from src.training.TripletEmbedTrain import TripletEmbeddModel
from src.utility import utils
from pathlib import Path
from PIL import Image
from config import settings
import numpy as np
import chromadb
import torch

class VectorDataAccess():
    def __init__(self, dbPath, dbName):
        self.client = chromadb.PersistentClient(path=dbPath)
        self.collection = self.client.get_or_create_collection(
            name=dbName,
            metadata={"hnsw:space": "l2"} #vector distance
        )

    def upsert_Vector(self, vectors, paths:str):
        self.collection.upsert(
            ids=paths,
            embeddings=vectors
        )

    def getNearest_Vectors(self, vector, k=5):
        # results format below
        # {
        #     'ids': [['./data/custom/image2.tif', './data/custom/image1.tif']], 
        #     'distances': [[0.1234, 0.4567]], 
        #     'metadatas': [[None, None]], 
        #     'embeddings': None, 
        #     'documents': [[None, None]], 
        #     'uris': None, 
        #     'data': None
        # }
        results = self.collection.query(
            query_embeddings=vector,
            n_results=k 
        )
        return results
    
class EmbedModelDataAccess():
    def __init__(self, model:EmbeddingModel, dbName="Writer_Identifiers"):
        self.model = model
        utils.modelToGPU(self.model)
        self.vectorDB = VectorDataAccess(settings.VECTOR_DATA_DIR, dbName)
    def UpsertFolder(self, directory:str):
        print('Inserting images and vectors from ', directory)
        files = ImageLabels.getFolderFiles(directory)
        vectors_lst = self.model.get_list_embeddings(files)
    
        self.vectorDB.upsert_Vector(vectors_lst, files)

    def NearestImages(self, imageIn, k=5):
        image = utils.getRGBImage(imageIn)
        vectorTensor = self.model.get_embedding(image)
        embeddingVector = vectorTensor.detach().cpu().tolist()
        results = self.vectorDB.getNearest_Vectors(embeddingVector, k)
        utils.displayImages(results['ids'][0], k)
        return results
    



        


    


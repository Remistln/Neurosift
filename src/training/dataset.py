import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from src.collector.metadata_store import MetadataStore, ImageMetadata
from sqlalchemy.orm import sessionmaker
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuroSiftDataset(Dataset):
    def __init__(self, split="train", transform=None, target_classes=["T1", "T2", "FLAIR"]):
        self.transform = transform
        self.classes = target_classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load splits
        with open("data/splits.json", "r") as f:
            splits = json.load(f)
        
        target_patients = splits.get(split, [])
        logger.info(f"Init {split} set with {len(target_patients)} patients")

        # DB connection
        store = MetadataStore()
        Session = sessionmaker(bind=store.engine)
        session = Session()
        
        # Filter records
        self.records = session.query(ImageMetadata).filter(
            ImageMetadata.modality.in_(self.classes),
            ImageMetadata.pmc_id.in_(target_patients)
        ).all()
        
        session.close()
        
        # Build index
        self.data_index = []
        for r in self.records:
            self.data_index.append({
                "path": r.s3_key,
                "label": r.modality
            })
            
        logger.info(f"Loaded {len(self.data_index)} images")

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        item = self.data_index[idx]
        path = item["path"]
        label_str = item["label"]
        
        # Binary read for unicode support
        stream = np.fromfile(path, np.uint8)
        image = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        
        if image is None:
            raise FileNotFoundError(f"Bad image: {path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Augment
        if self.transform:
            image = self.transform(image)
            
        # Label
        label = self.class_to_idx[label_str]
        
        return image, label

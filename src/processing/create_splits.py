import json
import random
import os
from src.collector.metadata_store import MetadataStore, ImageMetadata

def create_splits():
    store = MetadataStore()
    session = store.Session()
    
    # Get all unique Patient IDs
    patients_query = session.query(ImageMetadata.pmc_id).distinct().all()
    patients = [p[0] for p in patients_query]
    
    # Shuffle and Split (80/20)
    random.seed(42) # Reproducibility
    random.shuffle(patients)
    
    split_idx = int(0.8 * len(patients))
    train_patients = patients[:split_idx]
    test_patients = patients[split_idx:]
    
    splits = {
        "train": train_patients,
        "test": test_patients
    }
    
    # Save to file
    with open("data/splits.json", "w") as f:
        json.dump(splits, f, indent=4)
        
    print(f"Split created: {len(train_patients)} Train Patients, {len(test_patients)} Test Patients.")
    session.close()

if __name__ == "__main__":
    create_splits()

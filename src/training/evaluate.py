import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from src.training.dataset import NeuroSiftDataset
from src.inference.predictor import ModelPredictor
from torch.utils.data import DataLoader
import os

def evaluate_model():
    # Load Dataset
    dataset = NeuroSiftDataset(target_classes=["T1", "T2", "FLAIR"])
    # No transform needed for raw pixel access if we use predictor, 
    # but predictor handles opening.
    # Actually, let's use the dataset's logical structure but let predictor handle inference to match prod.
    
    predictor = ModelPredictor()
    if not predictor.model:
        return

    y_true = []
    y_pred = []
    
    print(f"Evaluating on {len(dataset)} images...")
    
    for i in range(len(dataset)):
        item = dataset.data_index[i]
        path = item["path"]
        true_label = item["label"]
        
        # Predict
        res = predictor.predict(path)
        if res:
            y_true.append(true_label)
            y_pred.append(res['label'])
            
        if i % 100 == 0:
            print(f"Processed {i}...")

    # Confusion Matrix
    labels = ["T1", "T2", "FLAIR"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - NeuroSift Modality Classification')
    plt.savefig('confusion_matrix.png')
    print("Confusion Matrix saved to confusion_matrix.png")
    
    # Report
    report = classification_report(y_true, y_pred, target_names=labels)
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    evaluate_model()

# NeuroSift

NeuroSift est un pipeline complet de classification d'IRM cérébrales (T1, T2, FLAIR) conçu pour simuler un environnement de production clinique.
Il intègre l'ingestion de données médicales réelles, le traitement d'images DICOM, l'entraînement d'un modèle ResNet18 avec PyTorch, et un déploiement via Docker et Streamlit.

![Demo](demo_placeholder.png)

## Fonctionnalités Clés

*   **Ingestion de Données** : Téléchargement automatique et structuré depuis l'API The Cancer Imaging Archive (TCIA / UPENN-GBM).
*   **Traitement Médical** : Fenêtrage (Windowing), Normalisation et Gestion du format DICOM.
*   **Deep Learning** : Modèle ResNet18 (Transfer Learning) avec une précision de 99.3%.
*   **Rigueur Scientifique** : Validation stricte par patient (Patient-Level Split) pour éviter le data leakage.
*   **Infrastructure** : Architecture micro-services avec Docker Compose (App, MinIO, Postgres, MLflow).

## Stack Technique

*   **Langage** : Python 3.10
*   **Machine Learning** : PyTorch, Torchvision, Scikit-learn
*   **Data Engineering** : Pydicom, OpenCV, Pandas, SQLAlchemy
*   **DevOps** : Docker, MinIO (S3), MLflow
*   **Frontend** : Streamlit

## Installation & Démarrage

### Pré-requis
*   Docker & Docker Compose

### Lancement Rapide
1.  Cloner le repo :
    ```bash
    git clone https://github.com/Remistln/Neurosift.git
    cd neurosift
    ```

2.  Lancer l'application via Docker :
    ```bash
    docker-compose up --build
    ```

3.  Accéder au dashboard :
    *   **App** : `http://localhost:8501`
    *   **MinIO** : `http://localhost:9001`

## Résultats
Le modèle atteint 99% de précision sur les séquences anatomiques standards.
*Note : Une étude de cas intéressante sur les limites du modèle (T1+Contrast vs FLAIR) est documentée dans le code.*
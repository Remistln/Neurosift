import os
from dotenv import load_dotenv

load_dotenv()

# App Config
USE_LOCAL_STORAGE = os.getenv("USE_LOCAL_STORAGE", "True").lower() == "true"

# MinIO Config (if needed)
# MinIO Config (if needed)
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin") # Default docker values
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "neuro-images")

# Local Storage Config
LOCAL_DATA_DIR = os.path.join(os.getcwd(), "data", "raw")

# Database Config
# If using Docker/Postgres
DB_USER = os.getenv("POSTGRES_USER", "user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "password")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "neurosift")

# If using SQLite
SQLITE_DB_PATH = "neurosift.db"

# PubMed Config
# Always provide an email to NCBI so they can contact you if you flood them
EMAIL = os.getenv("NCBI_EMAIL", "your.email@example.com") 

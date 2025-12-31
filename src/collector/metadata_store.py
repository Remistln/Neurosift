import os
import logging
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from src.config import USE_LOCAL_STORAGE, DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME, SQLITE_DB_PATH, MINIO_BUCKET_NAME

logger = logging.getLogger(__name__)

Base = declarative_base()

class ImageMetadata(Base):
    __tablename__ = 'image_metadata'
    id = Column(Integer, primary_key=True, autoincrement=True)
    pmc_id = Column(String(50))
    figure_id = Column(String(50))
    graphic_id = Column(String(50))
    s3_key = Column(String(200))
    caption = Column(Text)
    modality = Column(String(50), nullable=True)
    pathology = Column(String(100), nullable=True)
    is_valid = Column(Integer, default=1) # 1=True, 0=False (SQLite doesn't have native Boolean)
    collected_at = Column(DateTime, default=datetime.utcnow)

class MetadataStore:
    def __init__(self):
        if USE_LOCAL_STORAGE:
            logger.info("Using SQLite Database (Local Mode)")
            db_url = f"sqlite:///{SQLITE_DB_PATH}"
        else:
            db_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def save_image_metadata(self, meta_dict):
        session = self.Session()
        try:
            exists = session.query(ImageMetadata).filter_by(
                pmc_id=meta_dict['pmc_id'], 
                graphic_id=meta_dict['graphic_id']
            ).first()
            
            if not exists:
                # Determine storage key prefix
                prefix = "local" if USE_LOCAL_STORAGE else MINIO_BUCKET_NAME
                
                new_record = ImageMetadata(
                    pmc_id=meta_dict['pmc_id'],
                    figure_id=meta_dict['fig_id'],
                    graphic_id=meta_dict['graphic_id'],
                    s3_key=f"{prefix}/{meta_dict['filename']}",
                    caption=meta_dict.get('caption', ''),
                    modality=None,
                    pathology=None
                )
                session.add(new_record)
                session.commit()
                logger.info(f"Saved metadata for {meta_dict['graphic_id']}")
            else:
                logger.info(f"Metadata already exists for {meta_dict['graphic_id']}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            session.rollback()
        finally:
            session.close()

import os
import pydicom
import numpy as np
import cv2
import logging
from src.config import LOCAL_DATA_DIR
from src.collector.metadata_store import MetadataStore, ImageMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DicomProcessor:
    def __init__(self):
        self.raw_dir = os.path.join(LOCAL_DATA_DIR, "dicom")
        self.processed_dir = os.path.join(LOCAL_DATA_DIR, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        self.store = MetadataStore()

    def apply_window(self, image, center=None, width=None):
        # MRI robust normalization (Percentile scaling)
        # Ignore zeros (background) for calculation
        if np.max(image) == 0:
            return image.astype(np.uint8)
            
        p1 = np.percentile(image[image > 0], 1)
        p99 = np.percentile(image[image > 0], 99)
        
        # Clip
        image = np.clip(image, p1, p99)
        
        # Min-Max Scaling to 0-255
        image = image - p1
        image = image / (p99 - p1 + 1e-8) # Avoid div/0
        image = (image * 255).astype(np.uint8)
        
        return image

    def read_dicom(self, path):
        try:
            ds = pydicom.dcmread(path)
            image = ds.pixel_array.astype(np.float32)
            
            # Rescale slope intercept
            slope = getattr(ds, 'RescaleSlope', 1)
            intercept = getattr(ds, 'RescaleIntercept', 0)
            image = (image * slope) + intercept
            
            return image, ds
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            return None, None

    def process_patient(self, patient_id):
        patient_dir = os.path.join(self.raw_dir, patient_id)
        images = []
        
        for root, dirs, files in os.walk(self.raw_dir):
            if patient_id not in root:
                continue
                
            for file in files:
                if file.endswith(".dcm"):
                    full_path = os.path.join(root, file)
                    img, ds = self.read_dicom(full_path)
                    
                    if img is not None:
                        # Brain window
                        windowed = self.apply_window(img, center=40, width=80)
                        
                        # Output path
                        out_name = f"{patient_id}_{ds.SeriesInstanceUID[-5:]}_{ds.InstanceNumber}.png"
                        out_path = os.path.join(self.processed_dir, out_name)
                        
                        # Write image
                        is_success, buffer = cv2.imencode(".png", windowed)
                        if is_success:
                            with open(out_path, "wb") as f:
                                f.write(buffer)
                            images.append(out_path)
                            logger.info(f"Saved {out_path}")
                        else:
                            logger.error(f"Failed to save {out_path}")
                        
        logger.info(f"Processed {len(images)} images for {patient_id}")

    def run(self):
        session = self.store.Session()
        count = 0 
        
        for root, dirs, files in os.walk(self.raw_dir):
            for file in files:
                if file.endswith(".dcm"):
                    count += 1
                    if count % 50 == 0:
                        logger.info(f"Processing image {count}...")
                        
                    path = os.path.join(root, file)
                    img, ds = self.read_dicom(path)
                    
                    if img is not None:
                        # Metadata
                        pid = getattr(ds, 'PatientID', 'Unknown')
                        sex = getattr(ds, 'PatientSex', 'Unknown')
                        age = getattr(ds, 'PatientAge', 'Unknown')
                        modality = getattr(ds, 'Modality', 'MR')
                        series_uid = getattr(ds, 'SeriesInstanceUID', 'Unknown')
                        
                        # Windowing
                        w_img = self.apply_window(img, 40, 80)
                        
                        # Save
                        out_filename = f"{pid}_{series_uid[-5:]}_{count}.png"
                        out_path = os.path.join(self.processed_dir, out_filename)
                        
                        is_success, buffer = cv2.imencode(".png", w_img)
                        if is_success:
                            with open(out_path, "wb") as f:
                                f.write(buffer)
                            
                            # DB record
                            meta = ImageMetadata(
                                pmc_id=pid,
                                graphic_id=out_filename,
                                s3_key=out_path,
                                modality=modality,
                                caption=f"Age: {age}, Sex: {sex}",
                                is_valid=1
                            )
                            session.add(meta)

        session.commit()
        session.close()
        logger.info(f"Total processed: {count}")

if __name__ == "__main__":
    proc = DicomProcessor()
    proc.run()

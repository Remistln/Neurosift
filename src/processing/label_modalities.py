import os
import pydicom
import logging
from src.collector.metadata_store import MetadataStore, ImageMetadata
from src.config import LOCAL_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModalityLabeler:
    def __init__(self):
        self.store = MetadataStore()
        # Heuristics
        self.rules = {
            "T1": ["t1", "T1"],
            "T2": ["t2", "T2"],
            "FLAIR": ["flair", "FLAIR", "Fluid"],
            "PERFUSION": ["perf", "Perfusion", "DSC", "DCE", "ep2d"],
            "DTI": ["DTI", "Diffusion"]
        }

    def infer_modality(self, description):
        if not description:
            return "Unknown"
        
        desc = description.upper()
        
        # Priority checks
        if "FLAIR" in desc:
            return "FLAIR"
            
        if "T1" in desc:
            return "T1"
            
        if "T2" in desc:
            return "T2"
            
        if "PERF" in desc or "EP2D" in desc:
            return "PERFUSION"
            
        if "DTI" in desc:
            return "DTI"
            
        return "Other"

    def run(self):
        session = self.store.Session()
        records = session.query(ImageMetadata).all()
        
        logger.info(f"Labeling {len(records)} records...")
        
        # Build map from raw files
        raw_dir = os.path.join(LOCAL_DATA_DIR, "dicom")
        series_map = {} 
        
        logger.info("Scanning raw data...")
        for root, dirs, files in os.walk(raw_dir):
            for file in files:
                if file.endswith(".dcm"):
                    try:
                        ds = pydicom.dcmread(os.path.join(root, file), stop_before_pixels=True)
                        uid = getattr(ds, 'SeriesInstanceUID', None)
                        desc = getattr(ds, 'SeriesDescription', '')
                        
                        if uid and uid not in series_map:
                            label = self.infer_modality(desc)
                            series_map[uid] = label
                    except:
                        pass
                        
        logger.info(f"Found {len(series_map)} unique series")
        
        # Update DB
        updated = 0
        for r in records:
            # Format: {pid}_{series_uid_suffix}_{count}.png
            # Match by UID suffix
            short_uid = r.graphic_id.split('_')[1]
            
            found_label = "Unknown"
            for full_uid, label in series_map.items():
                if full_uid.endswith(short_uid):
                    found_label = label
                    break
            
            if found_label != "Unknown":
                r.modality = found_label
                updated += 1
                
        session.commit()
        logger.info(f"Updated {updated} records")
        session.close()

if __name__ == "__main__":
    lbl = ModalityLabeler()
    lbl.run()

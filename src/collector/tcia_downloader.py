import os
import logging
import pandas as pd
from tcia_utils import nbia
from src.config import LOCAL_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TCIADownloader:
    def __init__(self, collection="UPENN-GBM"):
        self.collection = collection
        self.raw_dir = os.path.join(LOCAL_DATA_DIR, "dicom")
        os.makedirs(self.raw_dir, exist_ok=True)

    def list_patients(self):
        """List patients in the collection."""
        logger.info(f"Fetching patients for {self.collection}...")
        # getPatient is tricky, let's use getSeries which we know works better
        # and extract unique PatientID from it.
        series_data = nbia.getSeries(collection=self.collection)
        
        # series_data is usually a list of dicts/json
        # Extraction
        if series_data and isinstance(series_data, list):
             patient_ids = list(set([s['PatientID'] for s in series_data]))
             # Return format compatible with previous code expects list of dicts or just ids?
             # Previous code: patient_ids = [p['PatientID'] for p in patients]
             # So let's return a list of dicts to match strictly or update download_cohort
             return [{'PatientID': pid} for pid in patient_ids]
        
        return []

    def download_cohort(self, num_patients=3):
        """Download DICOMs for a subset of patients."""
        patients = self.list_patients()
        
        # tcia_utils returns a list of dictionaries usually, or json
        # Let's inspect the first few
        patient_ids = [p['PatientID'] for p in patients]
        logger.info(f"Found {len(patient_ids)} patients. Downloading {num_patients}...")
        
        subset = patient_ids[:num_patients]
        
        for pid in subset:
            logger.info(f"Downloading data for {pid}...")
            # downloadSeries automatically handles folder structure usually
            # But we can organize it nicely.
            # nbia.downloadSeries allows filtering.
            
            # We specifically want MR (Magnetic Resonance)
            series = nbia.getSeries(collection=self.collection, patientId=pid, modality="MR")
            series_uids = [s['SeriesInstanceUID'] for s in series]
            
            logger.info(f"Found {len(series_uids)} MR series for {pid}")
            
            # Download
            # nbia.downloadSeries(SeriesInstanceUID=series_uids, path=self.raw_dir)
            # downloadSeries uses series_data (first arg)
            nbia.downloadSeries(series_data=series_uids, path=self.raw_dir, input_type="list")
            
        logger.info("Download completed.")

if __name__ == "__main__":
    downloader = TCIADownloader(collection="UPENN-GBM")
    downloader.download_cohort(num_patients=2) # Start very small (DICOMs are big)

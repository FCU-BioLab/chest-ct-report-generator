
import logging
from monai.bundle import download

logging.basicConfig(level=logging.INFO)

try:
    download(name="lung_nodule_ct_detection", bundle_dir="bundles")
    print("✅ Bundle downloaded successfully.")
except Exception as e:
    print(f"❌ Bundle download failed: {e}")

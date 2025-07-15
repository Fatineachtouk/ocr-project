
#C:\Users\pc\conda\envs\kraken\python.exe

import warnings
import logging
from PIL import Image
from kraken import binarization, pageseg, rpred
from kraken.lib.models import load_any

warnings.filterwarnings("ignore", category=UserWarning, module="kraken.rpred")

logging.basicConfig(
    filename="kraken_ocr.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def kraken_ocr(image_path: str, model_path: str) -> str:
    image = Image.open(image_path).convert('L')  # Load as grayscale
    bin_img = binarization.nlbin(image)
    segments = pageseg.segment(bin_img)
    model = load_any(model_path)
    predictions = rpred.rpred(model, bin_img, segments)
    return ' '.join([pred.prediction for pred in predictions])


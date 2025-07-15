import io
import os
import re
import cv2
import fitz
import clip
import torch
import paddle
import logging
import fasttext
import warnings
import requests
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from transformers import logging as hf_logging
from transformers import VisionEncoderDecoderModel, TrOCRProcessor


#C:\Users\pc\conda\envs\ocr-htr\python.exe

# Huggingface hub
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

hf_logging.set_verbosity_error()


# Suppress PaddleOCR logging
logging.getLogger('ppocr').setLevel(logging.ERROR)

if torch.cuda.is_available():
    device = "cuda"

elif paddle.is_compiled_with_cuda():
    device = "gpu"
# Default to CPU
else:
    device = "cpu"

# Loading the arabic model globally
ocr_model = PaddleOCR(use_angle_cls=True, lang='arabic')  # Arabic OCR model loaded once

model, preprocess = clip.load("ViT-B/32", device=device)   #Handwritten/ printed text classification model (CLIP)


if not os.path.exists("lid.176.bin"):
    raise FileNotFoundError("Download FastText model: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
fasttext_model = fasttext.load_model("lid.176.bin")


# Define the candidate labels
text_labels = ["handwritten text", "printed text"]
text = clip.tokenize(text_labels).to(device)

lang_labels = ["handwritten Arabic", "handwritten French", "handwritten English"]
text_lang = clip.tokenize(lang_labels).to(device)


def is_gpu_available():
    try:
        return paddle.is_compiled_with_cuda() and paddle.device.is_compiled_with_cuda()
    except Exception:
        return False

use_gpu = is_gpu_available()

def preprocessing(image):
    """
    Enhance image quality before OCR.
    Args:
        image (np.ndarray): Input image (BGR).
    Returns:
        np.ndarray: Enhanced image.
    """
    if image is None:
        raise ValueError("Empty image passed to preprocessing()")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 1. Contrast Enhancement
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=6)  # 2. Denoising

    # 3. Resize if needed
    h, w = denoised.shape
    if h < 600:
        scale = 800 / h
        processed = cv2.resize(denoised, (int(w * scale), 800), interpolation=cv2.INTER_CUBIC)
    elif h > 2500:
        scale = 1800 / h
        processed = cv2.resize(denoised, (int(w * scale), 1800), interpolation=cv2.INTER_AREA)
    else:
        processed = denoised

    return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)  # 4. Convert back to BGR


def is_clear(image):
    """
    Check if the image is sharp, bright, and noise-free for OCR.
    Args:
        image (np.ndarray): Input image.
    Returns:
        bool: True if the image is suitable for OCR.
    """
    # len(image.shape) == 3 means the image is colored
    # if the image is grayscale, we can directly use it by copying it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    
    # 1. Check sharpness using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Check contrast - more lenient for thermal receipts
    contrast = np.std(gray)
    
    # 3. Check brightness 
    brightness = np.mean(gray)
    
    # 4. Check noise - more tolerant
    kernel = np.ones((3, 3), np.float32) / 9
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    noise = np.mean(np.abs(gray.astype(np.float32) - local_mean))

    # 5. Blank page detection  
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    blank = np.max(hist) / gray.size

    return laplacian_var > 30 and contrast > 20 and 30 <= brightness <= 230 and noise < 50 and blank < 0.7


def is_pdf_selectable(pdf_path, min_text_threshold=50):
    """
    Determine if PDF contains selectable text.
    Args:
        pdf_path (str): Path to the PDF.
        min_text_threshold (int): Minimum characters to consider selectable.
    Returns:
        bool: True if PDF is selectable, False otherwise.
    """
    try:
        doc = fitz.open(pdf_path)
        total_chars = sum(len(re.sub(r'\s+', '', page.get_text())) for page in doc)
        pages_with_text = sum(1 for page in doc if len(re.sub(r'\s+', '', page.get_text())) > 10)
        total_pages = len(doc)
        doc.close()
        # Calculate coverage as the ratio of pages with text to total pages
        coverage = pages_with_text / total_pages if total_pages > 0 else 0

        return total_chars >= min_text_threshold and coverage >= 0.5
    except Exception as e:
        return {'is_selectable': False, 'error': str(e)}


def extract_text_pdf(pdf_path):
    """
    Extract text from each page of a selectable PDF.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        dict: {'text': str}
    """
    try:
        doc = fitz.open(pdf_path)
        all_text = [page.get_text() for page in doc]
        doc.close()
        return {'text': '\n'.join(all_text)}
    except Exception as e:
        return {'text': '', 'error': str(e)}


def pdf_to_images(pdf_path, dpi=300):
    """
    Convert PDF pages to images.
    Args:
        pdf_path (str): Path to PDF.
        dpi (int): Resolution for conversion.
    Returns:
        list: List of OpenCV images.
    """
    try:
        doc = fitz.open(pdf_path)
        images = []
        for page in doc:
            # Create transformation matrix for desired DPI
            mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 is default DPI

            # Render page as image
            pix = page.get_pixmap(matrix=mat)
            img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
            images.append(img)
        doc.close()
        return images
    except Exception:
        return []


def text_classifier(img):
    """
    Classify whether the text in the image is handwritten or printed.

    Args:
        img (str): The input image.

    Returns:
        str: Either 'handwritten' or 'printed'.
    """

    # Run the model
    with torch.no_grad():
       image_features = model.encode_image(img)
       text_features = model.encode_text(text)

    # Compute similarity
    logits_per_image, _ = model(img, text)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

    return text_labels[probs[0].argmax()].split()[0].lower()  #handwritten or printed


def hand_lang_detector(img):
    """
    Detect the language of handwritten text in an image.

    Args:
        image (str): the input image.

    Returns:
        str: Detected language - 'arabic', 'french', or 'english'.
    """

    with torch.no_grad():
        image_features = model.encode_image(img)
        text_features = model.encode_text(text_lang)
        logits_per_image, _ = model(img, text_lang)
        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

    return lang_labels[probs[0].argmax()].split()[-1].lower()  #return the language with the highest probability


def trocr(path):
    """
    Extract english handwritten text using TrOCR.
    Args:
        path (str): path of the input image
    Returns:
        str: extracted handwritten text
    """

    image = Image.open(path).convert("RGB")
    # Load processor and model
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")

    # Prepare image for model
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # Predict
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text


def detect_language(text):
    """
    Detect language from text using FastText.
    Args:
        text (str): Input text.
    Returns:
        str: Language code (lowercase).
    """
    try:
        label, _ = fasttext_model.predict(text.strip(), k=1)
        return label[0].replace("__label__", "").lower()
    except Exception:
        return 'en'


def ocr_img(img):
    """
    Perform OCR on a single image and return a list of text lines (strings).
    Args:
        img (np.ndarray): Input image.
    Returns:
        list[str]: List of recognized text lines.
    """
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("Invalid image input")

    # Initial OCR with Arabic model to extract text for detection
    result = ocr_model.ocr(img, cls=True)
    initial_lines = []
    for line in result:
        for word_info in line:
            if word_info and len(word_info) >= 2:
                initial_lines.append(word_info[1][0])

    # Detect language from text (optional, you can keep or remove this logic)
    detected_lang = detect_language(" ".join(initial_lines)) if initial_lines else 'en'

    if detected_lang == 'arabic':
        ocrr = PaddleOCR(
            use_angle_cls=True,
            use_gpu=use_gpu,
            rec_algorithm='SVTR_LCNet', 
            lang='ar',
            rec_model_dir='arabic_PP-OCRv3_rec_infer',
            rec_char_dict_path='arabic_dict.txt')  # path to extracted model
        refined_result = ocrr.ocr(img, cls=True)
  
    else:
        # Rerun OCR with the proper language model
        refined_ocr = PaddleOCR(use_angle_cls=True, lang=detected_lang)
        refined_result = refined_ocr.ocr(img, cls=True)
    
    
    lines = []
    for line in refined_result:
        for word_info in line:
            if word_info and len(word_info) >= 2:
                lines.append(word_info[1][0])

    return lines



def ocr(path):
    """
    Main OCR entry point for image or PDF.
    Args:
        path (str): Path to image or PDF.
    Returns:
        list[str]: List of recognized text lines.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {path}")

    extension = os.path.splitext(path)[1].lower()
    if extension == '.pdf':
        pdf_info = is_pdf_selectable(path)
        if pdf_info:
            result = extract_text_pdf(path)
            return result.get('text', '').splitlines()
        else:
            images = pdf_to_images(path)
            all_lines = []
            for img in images:
                if not is_clear(img):
                    img = preprocessing(img)
                result = ocr_img(img)  # Now returns list[str]
                all_lines.extend(result)
            return all_lines

    else:
        img = preprocess(Image.open(path)).unsqueeze(0).to(device)  
        if text_classifier(img)=='handwritten':
            if hand_lang_detector(img)=='english':
                return trocr(path)
            if hand_lang_detector(img)=='french':
                response = requests.post("http://127.0.0.1:5001/ocr", json={"image_path": path})

                if response.ok:
                    return response.json().get("text", "").splitlines()
                else:
                    raise RuntimeError(f"Kraken API error: {response.status_code} - {response.text}")
        else: #use paddle
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Could not read image from {path}")
            if not is_clear(img):
                img = preprocessing(img)
            return ocr_img(img)










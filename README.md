# OCR & HTR Project

## Overview

This is an OCR (Optical Character Recognition) and HTR (Handwritten Text Recognition) project that extracts printed or handwritten text from invoices and documents. Then an LLM model extracts the given metadata with descrition from the extracted text and returns a JSON format.

Multiple open-source models were used:

- **PaddleOCR** for printed text (English, French, Spanish)
- **arabic_PP-OCRv3_rec_infer** for Arabic text
- **TrOCR** for English handwritten text
- **Kraken** for French handwritten text
- **Fasttext** for language detection in extracted text.
- **Clip** for text classification (handwritten / printed) and language detection in handwritten text.
  
The project is built in two separate environments to avoid conflicts between dependencies:
- A main environment for the Paddle and TrOCR pipeline.
- A separate environment for Kraken ( kraken is used only if the text is french handwritten).

The main OCR & HTR pipeline communicates with the Kraken model through an HTTP API.

### Brief explanation of eaxh file :
- **ocr_pipeline.py**: All the functions used to extract the text, here's a visual of the pipeline:
- 
  <img width="1074" height="840" alt="Blank diagram" src="https://github.com/user-attachments/assets/c4fedffa-448a-4e55-bbc4-0745dc8f5cf7" />
- **process.py**: Once the text is extracted, it will be given to the LLM to extract certain metadata and classifies the type of the document.
- **french_htr.py**: Functions to extract handwritten french text.
- **kraken_app.py**: The kraken API.
- **app.py**: The main API to extract any kind of text( even if it handwritten and french since it calls the kraken API too).

---


## How to Use Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Fatineachtouk/ocr-project.git
cd ocr-project
```

### 2. Set Up the Main Environment
First Make sure you have python 3.10 installed!
if not, you can install it here : [python 3.10](https://www.python.org/downloads/release/python-3100/)
<details> <summary> Windows</summary>
  
```bash
python3.10 -m venv main_env
main_env\Scripts\activate
pip install -r requirements.txt
deactivate
```

</details> <details> <summary> macOS / Linux</summary>

```bash
python3.10 -m venv main_env
source main_env/bin/activate
pip install -r requirements.txt
deactivate
```
</details>

### 3. Set Up the Kraken Environment

<details> <summary> Windows</summary>
  
```bash
python3.10 -m venv kraken_env
kraken_env\Scripts\activate
pip install -r requirements-kraken.txt
```
</details> <details> <summary> macOS / Linux</summary>

```bash
python3.10 -m venv kraken_env
source kraken_env/bin/activate
pip install -r requirements-kraken.txt
```
</details>

## Download Language Identification File (FastText)
Download the [lid.176.bin](https://fasttext.cc/docs/en/language-identification.html) file from fastText and place it in the project folder (ocr-project).

## Running the APIs
You need to open two terminals, since there are two APIs.
### Kraken API
In the first terminal, run the fllowing:
```bash
# Activate kraken environment
# Windows:
kraken_env\Scripts\activate

# macOS/Linux:
source kraken_env/bin/activate

# Run Kraken server
python kraken_app.py
```
### Main OCR & HTR API
In the other terminal run the following :
```bash
# Activate kraken environment
# Windows:
main_env\Scripts\activate

# macOS/Linux:
source main_env/bin/activate

# Run Kraken server
python app.py
```
## Input Format (JSON)
This is the expected format for the main API:
```json
{
  "filePath": "",  
  "classificationOptions": [
    {
      "class": ["option1", "option2", "option3", "option4"],  
      "description": [
        "",
        "",
        "",
        ""
      ]
    },
    {
      "metadata": [
        { "fieldName": "option1", "description": "" },
        { "fieldName": "option2", "description": "" },
        { "fieldName": "", "description": "" },
        { "fieldName": "", "description": "" },
        { "fieldName": "", "description": "" }
      ]
    }
  ]
}
```



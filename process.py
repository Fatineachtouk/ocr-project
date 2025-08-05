import os
import re
import json
import textwrap
import ollama
from ocr_pipeline import ocr
from dotenv import load_dotenv

load_dotenv()

model = os.getenv('LLM_MODEL_NAME', 'gemma3:4b')

def query_ollama(prompt: str) -> str:
    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content']

def clean_json_response(text: str) -> str:
    return re.sub(r"```(?:json)?\n|\n```", "", text).strip()

def llm(input_data):
    # OCR
    lines = ocr(input_data['filePath'])
    if not isinstance(lines, (list, tuple)):
        lines = [str(lines)]
    raw_text = " ".join(lines).strip()
    formatted_text = textwrap.fill(raw_text, 80)

    # Prepare classification prompt
    classification_lines = []
    class_map = {}

    for option in input_data['classificationOptions']:
        class_name = option['class']
        description = option.get('description', '')
        classification_lines.append(f"- {class_name}: {description}")
        class_map[class_name] = option  # Store entire option for metadata later

    classification_prompt = f"""
You are a document classification expert.

Based on the OCR content below, choose the most appropriate class from the list.

OCR Content:
{formatted_text}

Classification Options:
{chr(10).join(classification_lines)}
 
Only return JSON in this format:
{{ "classification": "CIN" }}
""".strip()

    # Step 2: Run classification
    raw_class_response = query_ollama(classification_prompt)
    class_response = clean_json_response(raw_class_response)

    try:
        class_result = json.loads(class_response)
        predicted_class = class_result.get("classification", "").strip()
    except Exception:
        return {
            "error": "Failed to parse classification response",
            "raw_class_response": raw_class_response
        }

    if predicted_class not in class_map:
        return {
            "error": "Predicted class not in known classificationOptions",
            "predicted": predicted_class
        }

    # Step 3: Metadata extraction for the predicted class
    metadata = class_map[predicted_class].get("metadata", [])
    metadata_instructions = "\n".join([
        f"- {m['fieldName']}: {m['description']}" for m in metadata if m["fieldName"]
    ])

    metadata_prompt = f"""
You are an expert in document information extraction.

Based on the OCR content below, extract the following fields.

OCR Content:
{formatted_text}

Metadata Fields to Extract:
{metadata_instructions}

Return only JSON in this format:
{{
  "metadata": [
    {{ "fieldName": "full_name", "value": "..." }},
    ...
  ]
}}
""".strip()

    raw_metadata_response = query_ollama(metadata_prompt)
    metadata_response = clean_json_response(raw_metadata_response)

    try:
        metadata_result = json.loads(metadata_response)
        metadata = metadata_result.get("metadata", [])
    except Exception:
        return {
            "classification": predicted_class,
            "error": "Failed to parse metadata response",
            "raw_metadata_response": raw_metadata_response
        }

    return {
        "classification": predicted_class,
        "content": raw_text,
        "metadata": metadata
    }

import json
import re
import textwrap
from ocr_pipeline import ocr
import ollama

def query_ollama(prompt: str) -> str:
    response = ollama.chat(
        model='gemma3:4b', # gemma3:4b-it-qat is the quantisized model for efficiency
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content']

def clean_json_response(text: str) -> str:
    return re.sub(r"```(?:json)?\n|\n```", "", text).strip()

def llm(input_data):
    lines = ocr(input_data['filePath'])

    if not isinstance(lines, (list, tuple)):
      lines = [str(lines)]

    raw_text = " ".join(lines).strip()

    classification_block = ""
    classification_classes = input_data['classificationOptions'][0]['class']
    classification_desc = input_data['classificationOptions'][0]['description']
    metadata_fields = input_data['classificationOptions'][1]['metadata']

    metadata_part = "\n".join([f"- {f['fieldName']}: {f['description']}" for f in metadata_fields])

    if len(classification_classes) > 1:
        classification_block = "\nClassification Options:\n" + "\n".join([
            f"- {c}: {d}" for c, d in zip(classification_classes, classification_desc)
        ])

    prompt = f"""
You are a meticulous document analyzer.
Ignore police signature, security terms, expiration dates unless asked. Focus only on the descriptions.
Correct any OCR spelling issues if detected.
Return a valid, clean JSON with 3 keys:
- "content": the cleaned and corrected OCR content
- "classification": the best matching class (if provided)
- "metadata": a list of objects with "fieldName" and "value" (even if null)

Language Rules:
- If two or more languages are present, prefer French if found.
- If one language dominates, use it for the output.
- OCR text is line-based; order matters.
- First/last names may be uppercase.

OCR Content:
{textwrap.fill(raw_text, 80)}
{classification_block}

Metadata Fields to Extract:
{metadata_part}
""".strip()

    # Call LLM
    response = query_ollama(prompt)
    cleaned_response = clean_json_response(response)

    try:
        parsed = json.loads(cleaned_response)
        if "classification" in parsed and isinstance(parsed["classification"], str):
            classification = parsed["classification"]
            classification = classification.split(':')[0].strip()
            classification = classification.split('(')[0].strip()
            parsed["classification"] = classification
    except json.JSONDecodeError:
        parsed = {"error": "Failed to parse LLM response", "raw": response}

    return parsed



"""Task 1: Extract data from claim documents using Azure Document Intelligence."""

import argparse
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


DEFAULT_MODEL_ID = "prebuilt-document"
DEFAULT_API_VERSION = "2023-07-31"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value.rstrip("/")


def detect_content_type(file_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type or "application/octet-stream"


def analyze_document(
    file_path: Path,
    endpoint: str,
    api_key: str,
    model_id: str = DEFAULT_MODEL_ID,
    api_version: str = DEFAULT_API_VERSION,
    poll_interval: int = 3,
    timeout_seconds: int = 120,
) -> Dict[str, Any]:
    url = f"{endpoint}/formrecognizer/documentModels/{model_id}:analyze?api-version={api_version}"
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": detect_content_type(file_path),
    }

    with file_path.open("rb") as handle:
        response = requests.post(url, headers=headers, data=handle, timeout=60)

    response.raise_for_status()

    operation_location = response.headers.get("operation-location")
    if not operation_location:
        raise RuntimeError("Document Intelligence did not return operation-location.")

    deadline = time.time() + timeout_seconds
    poll_headers = {"Ocp-Apim-Subscription-Key": api_key}

    while time.time() < deadline:
        poll_response = requests.get(operation_location, headers=poll_headers, timeout=60)
        poll_response.raise_for_status()
        result = poll_response.json()
        status = result.get("status")
        if status == "succeeded":
            return result
        if status == "failed":
            raise RuntimeError(json.dumps(result, indent=2))
        time.sleep(poll_interval)

    raise TimeoutError("Timed out waiting for document analysis to complete.")


def collect_lines(result: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for page in result.get("analyzeResult", {}).get("pages", []):
        for line in page.get("lines", []):
            text = line.get("content", "").strip()
            if text:
                lines.append(text)
    return lines


def first_group(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    return match.group(match.lastindex or 0).strip()


def parse_claim_fields(lines: List[str]) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    date_pattern = r"\b(\d{2}[-/]\d{2}[-/]\d{4}|\d{4}[-/]\d{2}[-/]\d{2})\b"
    amount_pattern = r"((?:INR|USD)?\s?\d[\d,]*(?:\.\d{2})?|\$\s?\d[\d,]*(?:\.\d{2})?)"

    for index, text in enumerate(lines):
        if "Name" not in fields:
            name = first_group(r"^name\s*:?\s*(.+)$", text)
            if name:
                fields["Name"] = name
            elif re.fullmatch(r"name", text, re.IGNORECASE) and index + 1 < len(lines):
                next_line = lines[index + 1].strip()
                if next_line:
                    fields["Name"] = next_line

        if "Policy Number" not in fields:
            policy_number = first_group(r"policy\s*number\s*:?\s*(.+)$", text)
            if policy_number:
                fields["Policy Number"] = policy_number
            elif re.fullmatch(r"policy\s*number", text, re.IGNORECASE) and index + 1 < len(lines):
                fields["Policy Number"] = lines[index + 1].strip()

        if "Claim Amount" not in fields:
            claim_amount = first_group(r"(?:claim\s*amount|amount|amt)\s*:?\s*(.+)$", text)
            if claim_amount:
                fields["Claim Amount"] = claim_amount
            else:
                match = re.search(amount_pattern, text)
                if match:
                    fields["Claim Amount"] = match.group(1)

        if "Date" not in fields:
            match_date = re.search(date_pattern, text)
            if match_date:
                fields["Date"] = match_date.group(1)

    return fields


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1: Extract claim data with Azure Document Intelligence.")
    parser.add_argument("file", type=Path, help="Path to the claim form PDF/image.")
    parser.add_argument("--output", type=Path, help="Optional output path for raw JSON result.")
    parser.add_argument("--fields-output", type=Path, help="Optional output path for extracted fields JSON.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--api-version", default=DEFAULT_API_VERSION)
    args = parser.parse_args()

    endpoint = require_env("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    api_key = require_env("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    result = analyze_document(
        file_path=args.file,
        endpoint=endpoint,
        api_key=api_key,
        model_id=args.model_id,
        api_version=args.api_version,
    )
    fields = parse_claim_fields(collect_lines(result))

    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.fields_output:
        args.fields_output.write_text(json.dumps(fields, indent=2), encoding="utf-8")

    print(json.dumps(fields, indent=2))


if __name__ == "__main__":
    main()

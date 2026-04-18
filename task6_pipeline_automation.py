"""Task 6: Run the end-to-end flow from document extraction to prediction."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def run_python(script: str, *args: str) -> Dict[str, Any]:
    command = [sys.executable, script, *args]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    output = completed.stdout.strip()
    return json.loads(output) if output else {}


def maybe_predict(extracted_fields_path: Path) -> Optional[Dict[str, Any]]:
    scoring_uri = os.getenv("AZURE_ML_SCORING_URI")
    endpoint_key = os.getenv("AZURE_ML_ENDPOINT_KEY")
    if not scoring_uri or not endpoint_key:
        return None

    fields = json.loads(extracted_fields_path.read_text(encoding="utf-8"))
    raw_amount = str(fields.get("Claim Amount", "0"))
    cleaned_amount = raw_amount.replace(",", "").replace("$", "").replace("INR", "").replace("USD", "").strip()

    request_payload = {
        "data": [
            {
                "Claim_Amount": float(cleaned_amount or 0),
                "Claim_Type": fields.get("Claim Type", "General"),
                "Location": fields.get("Location", "Unknown"),
                "Previous_Claims": int(fields.get("Previous Claims", 0) or 0),
            }
        ]
    }

    request_path = extracted_fields_path.with_name("prediction_request.json")
    request_path.write_text(json.dumps(request_payload, indent=2), encoding="utf-8")
    return run_python(
        "task4_deploy_endpoint.py",
        "--invoke",
        "--scoring-uri",
        scoring_uri,
        "--endpoint-key",
        endpoint_key,
        "--request-json",
        str(request_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 6: End-to-end automation for extraction, search, and prediction.")
    parser.add_argument("document", type=Path)
    parser.add_argument("--workdir", type=Path, default=Path("outputs"))
    parser.add_argument("--index-name", default="claims-index")
    parser.add_argument("--search-filter", default="ClaimAmount gt 50000")
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    raw_output = args.workdir / "document_analysis.json"
    fields_output = args.workdir / "extracted_fields.json"

    extracted_fields = run_python(
        "task1_document_intelligence.py",
        str(args.document),
        "--output",
        str(raw_output),
        "--fields-output",
        str(fields_output),
    )

    if "Location" not in extracted_fields:
        extracted_fields["Location"] = "Unknown"
        fields_output.write_text(json.dumps(extracted_fields, indent=2), encoding="utf-8")

    search_upload = run_python(
        "task2_ai_search.py",
        "--index-name",
        args.index_name,
        "--upload-json",
        str(fields_output),
        "--search-text",
        "*",
        "--filter",
        args.search_filter,
    )

    prediction = maybe_predict(fields_output)
    result = {
        "document_json_path": str(raw_output),
        "structured_data_path": str(fields_output),
        "search_result": search_upload,
        "prediction_result": prediction,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

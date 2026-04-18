"""Task 2: Create an Azure AI Search index, upload claim data, and run queries."""

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List

import requests


DEFAULT_INDEX_NAME = "claims-index"
DEFAULT_API_VERSION = "2024-07-01"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value.rstrip("/")


def search_endpoint() -> str:
    service = require_env("AZURE_SEARCH_SERVICE")
    if service.startswith("https://"):
        return service
    return f"https://{service}.search.windows.net"


def admin_headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "api-key": require_env("AZURE_SEARCH_ADMIN_KEY"),
    }


def query_headers() -> Dict[str, str]:
    key = os.getenv("AZURE_SEARCH_QUERY_KEY") or require_env("AZURE_SEARCH_ADMIN_KEY")
    return {
        "Content-Type": "application/json",
        "api-key": key,
    }


def create_index(index_name: str, api_version: str) -> Dict[str, Any]:
    url = f"{search_endpoint()}/indexes/{index_name}?api-version={api_version}"
    payload = {
        "name": index_name,
        "fields": [
            {"name": "ClaimID", "type": "Edm.String", "key": True, "filterable": True, "retrievable": True},
            {"name": "Name", "type": "Edm.String", "searchable": True, "retrievable": True},
            {"name": "PolicyNumber", "type": "Edm.String", "searchable": True, "filterable": True, "retrievable": True},
            {"name": "ClaimAmount", "type": "Edm.Double", "filterable": True, "sortable": True, "retrievable": True},
            {"name": "Date", "type": "Edm.String", "filterable": True, "sortable": True, "retrievable": True},
            {"name": "Location", "type": "Edm.String", "searchable": True, "filterable": True, "facetable": True, "retrievable": True},
            {"name": "ClaimType", "type": "Edm.String", "searchable": True, "filterable": True, "facetable": True, "retrievable": True},
            {"name": "PreviousClaims", "type": "Edm.Int32", "filterable": True, "sortable": True, "retrievable": True},
            {"name": "FraudFlag", "type": "Edm.Boolean", "filterable": True, "facetable": True, "retrievable": True},
            {"name": "RawExtractedJson", "type": "Edm.String", "searchable": False, "retrievable": True},
        ],
    }
    response = requests.put(url, headers=admin_headers(), json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def normalize_document(data: Dict[str, Any]) -> Dict[str, Any]:
    amount_raw = str(data.get("Claim Amount") or data.get("ClaimAmount") or "0")
    cleaned_amount = amount_raw.replace(",", "").replace("$", "").replace("INR", "").replace("USD", "").strip()
    amount = float(cleaned_amount or 0)
    return {
        "@search.action": "mergeOrUpload",
        "ClaimID": str(data.get("ClaimID") or f"CLM-{uuid.uuid4().hex[:12].upper()}"),
        "Name": data.get("Name", ""),
        "PolicyNumber": data.get("PolicyNumber") or data.get("Policy Number", ""),
        "ClaimAmount": amount,
        "Date": data.get("Date", ""),
        "Location": data.get("Location", "Unknown"),
        "ClaimType": data.get("ClaimType", "General"),
        "PreviousClaims": int(data.get("PreviousClaims", 0) or 0),
        "FraudFlag": bool(data.get("FraudFlag", False)),
        "RawExtractedJson": json.dumps(data, ensure_ascii=True),
    }


def upload_documents(index_name: str, documents: List[Dict[str, Any]], api_version: str) -> Dict[str, Any]:
    url = f"{search_endpoint()}/indexes/{index_name}/docs/index?api-version={api_version}"
    payload = {"value": [normalize_document(doc) for doc in documents]}
    response = requests.post(url, headers=admin_headers(), json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def run_query(index_name: str, query: Dict[str, Any], api_version: str) -> Dict[str, Any]:
    url = f"{search_endpoint()}/indexes/{index_name}/docs/search?api-version={api_version}"
    response = requests.post(url, headers=query_headers(), json=query, timeout=60)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2: Create, load, and query an Azure AI Search index.")
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    parser.add_argument("--api-version", default=DEFAULT_API_VERSION)
    parser.add_argument("--create-index", action="store_true")
    parser.add_argument("--upload-json", type=Path, help="JSON file containing one document or a list of documents.")
    parser.add_argument("--search-text", default="*")
    parser.add_argument("--filter")
    parser.add_argument("--semantic", action="store_true")
    args = parser.parse_args()

    result: Dict[str, Any] = {}
    if args.create_index:
        result["index"] = create_index(args.index_name, args.api_version)

    if args.upload_json:
        loaded = json.loads(args.upload_json.read_text(encoding="utf-8"))
        documents = loaded if isinstance(loaded, list) else [loaded]
        result["upload"] = upload_documents(args.index_name, documents, args.api_version)

    query = {"search": args.search_text, "count": True}
    if args.filter:
        query["filter"] = args.filter
    if args.semantic:
        query["queryType"] = "semantic"
    result["query"] = run_query(args.index_name, query, args.api_version)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

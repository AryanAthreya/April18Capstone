"""Task 4: Prepare score.py, deploy the model, and invoke the endpoint."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from typing import Any, Dict, Optional

import requests


SCORE_SCRIPT = dedent(
    """
    import json
    import os
    from pathlib import Path

    import joblib
    import pandas as pd

    model = None


    def init():
        global model
        model_dir = Path(os.getenv("AZUREML_MODEL_DIR", "."))
        model_path = model_dir / "model.pkl"
        model = joblib.load(model_path)


    def run(raw_data):
        payload = json.loads(raw_data)
        rows = payload["data"]
        results = []
        for row in rows:
            frame = pd.DataFrame([
                {
                    "Claim_Amount": row.get("Claim_Amount", 0),
                    "Claim_Type": row.get("Claim_Type", "General"),
                    "Location": row.get("Location", "Unknown"),
                    "Previous_Claims": row.get("Previous_Claims", 0),
                }
            ])
            prediction = model.predict(frame)[0]
            probability = None
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(frame)[0][1])
            results.append({"prediction": int(prediction), "fraud_probability": probability})
        return results
    """
).strip() + "\n"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def write_scoring_assets(model_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "score.py"
    score_path.write_text(SCORE_SCRIPT, encoding="utf-8")
    target_model = output_dir / "model.pkl"
    target_model.write_bytes(model_path.read_bytes())
    return score_path


def deploy_endpoint(model_path: Path, endpoint_name: str, deployment_name: str, instance_type: str) -> Dict[str, Any]:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import (
        CodeConfiguration,
        Environment,
        ManagedOnlineDeployment,
        ManagedOnlineEndpoint,
        Model,
    )
    from azure.identity import DefaultAzureCredential

    subscription_id = require_env("AZURE_SUBSCRIPTION_ID")
    resource_group = require_env("AZURE_RESOURCE_GROUP")
    workspace_name = require_env("AZURE_ML_WORKSPACE")

    ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace_name)

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        write_scoring_assets(model_path, temp_path)

        model_asset = ml_client.models.create_or_update(
            Model(name=f"{endpoint_name}-model", path=str(model_path), type="custom_model")
        )

        environment = ml_client.environments.create_or_update(
            Environment(
                name=f"{endpoint_name}-env",
                description="Runtime for insurance fraud inference",
                image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
                conda_file={
                    "name": "fraud-inference",
                    "channels": ["conda-forge"],
                    "dependencies": [
                        "python=3.10",
                        "pip",
                        {"pip": ["azureml-inference-server-http", "joblib", "pandas", "scikit-learn"]},
                    ],
                },
            )
        )

        endpoint = ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key")
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()

        deployment = ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=model_asset.id,
            environment=environment.id,
            code_configuration=CodeConfiguration(code=str(temp_path), scoring_script="score.py"),
            instance_type=instance_type,
            instance_count=1,
        )
        ml_client.online_deployments.begin_create_or_update(deployment).result()
        ml_client.online_endpoints.begin_update(
            ManagedOnlineEndpoint(name=endpoint_name, traffic={deployment_name: 100})
        ).result()

        online_endpoint = ml_client.online_endpoints.get(endpoint_name)
        keys = ml_client.online_endpoints.get_keys(endpoint_name)
        return {
            "endpoint_name": endpoint_name,
            "deployment_name": deployment_name,
            "scoring_uri": online_endpoint.scoring_uri,
            "primary_key": keys.primary_key,
        }


def invoke_endpoint(scoring_uri: str, api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = requests.post(scoring_uri, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4: Deploy the fraud model to an Azure ML online endpoint.")
    parser.add_argument("--model-path", type=Path, default=Path("model.pkl"))
    parser.add_argument("--write-score-only", type=Path, help="Write score.py and model.pkl to a folder without deploying.")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--endpoint-name", default="insurance-claims-endpoint")
    parser.add_argument("--deployment-name", default="blue")
    parser.add_argument("--instance-type", default="Standard_DS3_v2")
    parser.add_argument("--invoke", action="store_true")
    parser.add_argument("--scoring-uri")
    parser.add_argument("--endpoint-key")
    parser.add_argument("--request-json", type=Path)
    args = parser.parse_args()

    result: Dict[str, Any] = {}

    if args.write_score_only:
        score_path = write_scoring_assets(args.model_path, args.write_score_only)
        result["score_script"] = str(score_path)

    if args.deploy:
        result["deployment"] = deploy_endpoint(
            model_path=args.model_path,
            endpoint_name=args.endpoint_name,
            deployment_name=args.deployment_name,
            instance_type=args.instance_type,
        )

    if args.invoke:
        if not args.scoring_uri or not args.endpoint_key or not args.request_json:
            raise RuntimeError("--invoke requires --scoring-uri, --endpoint-key, and --request-json.")
        payload = json.loads(args.request_json.read_text(encoding="utf-8"))
        result["prediction"] = invoke_endpoint(args.scoring_uri, args.endpoint_key, payload)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Task 5: Set up model monitoring and summarize drift actions."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_monitor_payload(endpoint_name: str, deployment_name: str, compute_name: str, target_data_uri: str) -> Dict[str, Any]:
    return {
        "display_name": f"{endpoint_name}-monitor",
        "endpoint_name": endpoint_name,
        "deployment_name": deployment_name,
        "compute_name": compute_name,
        "monitoring_target": {
            "ml_task": "classification",
            "target_data": {"path": target_data_uri},
        },
        "signals": {
            "data_drift": {
                "type": "data_drift",
                "metric_thresholds": {"jensen_shannon_distance": 0.1},
            },
            "prediction_drift": {
                "type": "prediction_drift",
                "metric_thresholds": {"jensen_shannon_distance": 0.1},
            },
        },
    }


def create_monitor(endpoint_name: str, deployment_name: str, compute_name: str, target_data_uri: str) -> Dict[str, Any]:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import MonitorDefinition
    from azure.identity import DefaultAzureCredential

    subscription_id = require_env("AZURE_SUBSCRIPTION_ID")
    resource_group = require_env("AZURE_RESOURCE_GROUP")
    workspace_name = require_env("AZURE_ML_WORKSPACE")

    ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace_name)

    payload = build_monitor_payload(endpoint_name, deployment_name, compute_name, target_data_uri)
    monitor = MonitorDefinition(**payload)
    created = ml_client.monitors.begin_create_or_update(monitor).result()
    return {"name": created.name, "provisioning_state": getattr(created, "provisioning_state", "unknown")}


def explain_monitoring() -> Dict[str, Any]:
    return {
        "what_happens_if_distribution_changes": [
            "Input features can drift away from the training distribution, which often lowers accuracy and precision.",
            "Prediction drift can indicate the model is seeing a new population or policy change.",
            "A monitor should alert you, trigger retraining, and prompt review of feature engineering and thresholds.",
        ],
        "recommended_actions": [
            "Store baseline training data and compare it to production inference data regularly.",
            "Alert on drift thresholds and route alerts to email, Teams, or Azure Monitor.",
            "Retrain the model through the pipeline in task 6 when drift or performance degradation is confirmed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 5: Configure Azure ML model monitoring.")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--deployment-name", default="blue")
    parser.add_argument("--compute-name", default="cpu-cluster")
    parser.add_argument("--target-data-uri", help="Azure ML data asset URI or storage path for production data.")
    parser.add_argument("--create-monitor", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result: Dict[str, Any] = {"guidance": explain_monitoring()}
    if args.create_monitor:
        if not args.target_data_uri:
            raise RuntimeError("--create-monitor requires --target-data-uri.")
        result["monitor"] = create_monitor(
            endpoint_name=args.endpoint_name,
            deployment_name=args.deployment_name,
            compute_name=args.compute_name,
            target_data_uri=args.target_data_uri,
        )

    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

import mlflow
import os
from datetime import datetime

# Set up MLflow
mlflow.set_tracking_uri("file:./mlruns")

# Log code adapter
mlflow.set_experiment("ds-assistant-code-adapter")
with mlflow.start_run(run_name="code_adapter_v1.0_existing"):
    mlflow.set_tags({
        'version': 'v1.0',
        'adapter_type': 'code',
        'status': 'production',
        'training_date': '2024-09-10',
        'model_family': 'mistral-7b'
    })
    mlflow.log_artifacts('output/adapters/', 'code_adapter')
    print("Code adapter logged to MLflow")

# Log theory adapter
mlflow.set_experiment("ds-assistant-theory-adapter")
with mlflow.start_run(run_name="theory_adapter_v1.0_existing"):
    mlflow.set_tags({
        'version': 'v1.0',
        'adapter_type': 'theory',
        'status': 'production',
        'training_date': '2024-09-10',
        'model_family': 'mistral-7b'
    })
    mlflow.log_artifacts('output_theory/adapters/', 'theory_adapter')
    print("Theory adapter logged to MLflow")

print("All existing models logged successfully!")

# airflow-champion-challenger-retraining

Apache Airflow project for **scheduled ML retraining** with **champion vs challenger evaluation**, **lightweight drift checks**, and **gated auto-promotion**.

## What it does

This repo demonstrates three core MLOps behaviors:

- Retrain on a schedule (default: monthly)
- Evaluate a newly trained challenger against the current champion
- Promote the challenger only if it meets performance criteria

## Where the DAG is

- `dags/model_retraining_dag.py`

## Configuration

Key environment variables:

- `MODEL_DIR` (default: `/opt/airflow/models`)
- `DATA_DB` (default: `/opt/airflow/data/hospital_capacity.db`)
- `AUC_IMPROVEMENT_THRESHOLD` (default: `0.01`)
- `RECALL_REGRESSION_THRESHOLD` (default: `0.10`)
- `DRIFT_KS_THRESHOLD` (default: `0.01`)
- `DRIFT_PSI_THRESHOLD` (default: `0.25`)

## Notes

This implementation stores artifacts locally for demo purposes. For multi-worker deployments, use shared storage (NFS/EFS) or swap to an object store/model registry (S3+MLflow, etc.).

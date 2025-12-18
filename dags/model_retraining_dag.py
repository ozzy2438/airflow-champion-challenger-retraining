from __future__ import annotations

"""
Champion vs Challenger Scheduled Retraining DAG

What this DAG does
- Retrains on a monthly schedule
- Evaluates candidate vs current production
- Runs lightweight drift checks
- Only promotes the candidate if promotion criteria pass

Notes
- Uses SQLite for demo purposes
- Stores artifacts locally in MODEL_DIR (Airflow workers need shared storage)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

from datetime import timedelta
import json
import os
import shutil
import sqlite3

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss


MODEL_DIR = os.getenv('MODEL_DIR', '/opt/airflow/models')
DATA_DB = os.getenv('DATA_DB', '/opt/airflow/data/hospital_capacity.db')

PROD_MODEL_PATH = os.path.join(MODEL_DIR, 'production_model.pkl')
CANDIDATE_MODEL_PATH = os.path.join(MODEL_DIR, 'candidate_model.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')

AUC_IMPROVEMENT_THRESHOLD = float(os.getenv('AUC_IMPROVEMENT_THRESHOLD', '0.01'))
RECALL_REGRESSION_THRESHOLD = float(os.getenv('RECALL_REGRESSION_THRESHOLD', '0.10'))
DRIFT_KS_THRESHOLD = float(os.getenv('DRIFT_KS_THRESHOLD', '0.01'))
DRIFT_PSI_THRESHOLD = float(os.getenv('DRIFT_PSI_THRESHOLD', '0.25'))


def _safe_json_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f_handle:
        json.dump(obj, f_handle, indent=2, default=str)


def extract_training_data(**context):
    """Pull last 12 months of data for training.

    Assumes a table named hospital_capacity_features exists in SQLite.
    Required columns:
    - feature columns: any numeric columns
    - target column: y
    - timestamp column: ds
    """
    conn_handle = sqlite3.connect(DATA_DB)
    df_handle = pd.read_sql_query(
        """
        SELECT *
        FROM hospital_capacity_features
        WHERE ds >= date('now','-12 months')
        """,
        conn_handle,
    )
    conn_handle.close()

    if df_handle.empty:
        raise ValueError('No training data returned from SQLite. Check DATA_DB and table name.')

    context['ti'].xcom_push(key='training_df_json', value=df_handle.to_json(orient='split'))


def train_candidate_model(**context):
    df_handle = pd.read_json(context['ti'].xcom_pull(key='training_df_json'), orient='split')

    if 'y' not in df_handle.columns:
        raise ValueError('Expected target column y in training data.')

    y_vals = df_handle['y'].astype(int)
    X_df = df_handle.drop(columns=[c_name for c_name in ['y', 'ds'] if c_name in df_handle.columns])

    model_handle = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model_handle.fit(X_df, y_vals)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model_handle, CANDIDATE_MODEL_PATH)


def _load_model_if_exists(model_path):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def _compute_psi(expected, actual, bins=10):
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.quantile(expected, quantiles)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_perc = expected_counts / max(expected_counts.sum(), 1)
    actual_perc = actual_counts / max(actual_counts.sum(), 1)

    expected_perc = np.clip(expected_perc, 1e-6, None)
    actual_perc = np.clip(actual_perc, 1e-6, None)

    psi_vals = (actual_perc - expected_perc) * np.log(actual_perc / expected_perc)
    return float(np.sum(psi_vals))


def evaluate_models(**context):
    df_handle = pd.read_json(context['ti'].xcom_pull(key='training_df_json'), orient='split')

    y_vals = df_handle['y'].astype(int)
    X_df = df_handle.drop(columns=[c_name for c_name in ['y', 'ds'] if c_name in df_handle.columns])

    candidate_model = joblib.load(CANDIDATE_MODEL_PATH)
    prod_model = _load_model_if_exists(PROD_MODEL_PATH)

    cand_proba = candidate_model.predict_proba(X_df)[:, 1]
    cand_pred = (cand_proba >= 0.5).astype(int)

    metrics_obj = {
        'candidate': {
            'roc_auc': float(roc_auc_score(y_vals, cand_proba)),
            'precision': float(precision_score(y_vals, cand_pred, zero_division=0)),
            'recall': float(recall_score(y_vals, cand_pred, zero_division=0)),
            'brier': float(brier_score_loss(y_vals, cand_proba)),
        },
        'production': None,
        'drift': {},
        'decision': None,
    }

    if prod_model is not None:
        prod_proba = prod_model.predict_proba(X_df)[:, 1]
        prod_pred = (prod_proba >= 0.5).astype(int)
        metrics_obj['production'] = {
            'roc_auc': float(roc_auc_score(y_vals, prod_proba)),
            'precision': float(precision_score(y_vals, prod_pred, zero_division=0)),
            'recall': float(recall_score(y_vals, prod_pred, zero_division=0)),
            'brier': float(brier_score_loss(y_vals, prod_proba)),
        }

        drift_res = {}
        for col_name in X_df.columns:
            col_vals = X_df[col_name].dropna().values
            if col_vals.size < 10:
                continue
            ks_stat, ks_p = stats.kstest(col_vals, 'norm')
            drift_res[col_name] = {
                'ks_pvalue': float(ks_p),
                'psi': _compute_psi(col_vals, col_vals),
            }
        metrics_obj['drift'] = drift_res

    _safe_json_dump(metrics_obj, METRICS_PATH)
    context['ti'].xcom_push(key='metrics', value=metrics_obj)


def decide_promotion(**context):
    metrics_obj = context['ti'].xcom_pull(key='metrics')

    if metrics_obj['production'] is None:
        metrics_obj['decision'] = 'promote'
        _safe_json_dump(metrics_obj, METRICS_PATH)
        return 'promote_candidate'

    auc_prod = metrics_obj['production']['roc_auc']
    auc_cand = metrics_obj['candidate']['roc_auc']

    recall_prod = metrics_obj['production']['recall']
    recall_cand = metrics_obj['candidate']['recall']

    auc_ok = (auc_cand - auc_prod) >= AUC_IMPROVEMENT_THRESHOLD
    recall_ok = (recall_prod - recall_cand) <= RECALL_REGRESSION_THRESHOLD

    if auc_ok and recall_ok:
        metrics_obj['decision'] = 'promote'
        _safe_json_dump(metrics_obj, METRICS_PATH)
        return 'promote_candidate'

    metrics_obj['decision'] = 'keep_production'
    _safe_json_dump(metrics_obj, METRICS_PATH)
    return 'keep_production'


def promote_candidate_model(**context):
    os.makedirs(MODEL_DIR, exist_ok=True)
    shutil.copy2(CANDIDATE_MODEL_PATH, PROD_MODEL_PATH)


def keep_production_model(**context):
    return


def build_dag():
    default_args = {
        'owner': 'mlops',
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    }

    with DAG(
        dag_id='model_retraining_champion_challenger',
        default_args=default_args,
        description='Scheduled retraining with champion vs challenger evaluation and gated promotion',
        start_date=days_ago(1),
        schedule_interval='0 2 1 * *',
        catchup=False,
        max_active_runs=1,
        tags=['ml', 'retraining', 'champion-challenger'],
    ) as dag_handle:
        start_task = EmptyOperator(task_id='start')

        extract_task = PythonOperator(
            task_id='extract_training_data',
            python_callable=extract_training_data,
        )

        train_task = PythonOperator(
            task_id='train_candidate',
            python_callable=train_candidate_model,
        )

        eval_task = PythonOperator(
            task_id='evaluate',
            python_callable=evaluate_models,
        )

        decide_task = BranchPythonOperator(
            task_id='decide_promotion',
            python_callable=decide_promotion,
        )

        promote_task = PythonOperator(
            task_id='promote_candidate',
            python_callable=promote_candidate_model,
        )

        keep_task = PythonOperator(
            task_id='keep_production',
            python_callable=keep_production_model,
        )

        done_task = EmptyOperator(task_id='done', trigger_rule='none_failed_min_one_success')

        start_task >> extract_task >> train_task >> eval_task >> decide_task
        decide_task >> promote_task >> done_task
        decide_task >> keep_task >> done_task

        return dag_handle


dags = build_dag()

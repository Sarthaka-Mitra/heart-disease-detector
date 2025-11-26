"""Generate evaluation metrics and plots for saved models.

Usage:
  python scripts/generate_evaluation_plots.py --input data/test_dataset.csv

This script:
- loads `models/scaler.pkl` and `models/feature_names.pkl` (if present)
- loads all `.pkl` model files from `models/` (except scaler/feature_names)
- prepares inputs using the same engineered features as the app by importing
  `prepare_batch_inputs` from `app.streamlit_app` (reuses app preprocessing logic)
- computes Accuracy, Precision, Recall, F1, ROC-AUC when `target` column exists
- writes `models/model_comparison.csv` and saves confusion matrix, ROC and
  feature importance plots under `reports/`

Notes:
- Input CSV must contain base columns expected by the app preprocessing.
- If the CSV includes a `target` column the script will compute metrics.
"""
from __future__ import annotations

import argparse
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
import logging


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_load_models(models_dir: str):
    files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    scaler = None
    feature_names = None
    models = {}
    for f in files:
        path = os.path.join(models_dir, f)
        if f == 'scaler.pkl':
            scaler = joblib.load(path)
            continue
        if f == 'feature_names.pkl':
            feature_names = joblib.load(path)
            continue
        # load all other pickles as model artifacts
        try:
            m = joblib.load(path)
            models[os.path.splitext(f)[0]] = m
        except Exception as e:
            logging.warning('Failed to load %s: %s', path, e)
    return models, scaler, feature_names


def plot_confusion_matrix(y_true, y_pred, labels, out_path: str, title: str | None = None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc(y_true, y_score, out_path: str, title: str | None = None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if title:
        plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_feature_importances(feature_names, importances, out_path: str, top_n: int = 20, title: str | None = None):
    df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    df['abs_imp'] = df['importance'].abs()
    df = df.sort_values('abs_imp', ascending=True).tail(top_n)
    plt.figure(figsize=(6, max(4, 0.25 * len(df))))
    plt.barh(df['feature'], df['importance'])
    plt.xlabel('Importance')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='data/test_dataset.csv', help='Labeled CSV to evaluate')
    parser.add_argument('--models-dir', '-m', default='models', help='Directory with model pickles')
    parser.add_argument('--out-dir', '-o', default='reports', help='Directory to write plots and metrics')
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    models, scaler, feature_names = safe_load_models(args.models_dir)
    if not models:
        print('No models found in', args.models_dir)
        return

    # Import preprocessing from the Streamlit app so we match exactly
    try:
        from app.streamlit_app import prepare_batch_inputs
    except Exception as e:
        print('Warning: failed to import prepare_batch_inputs from app.\n', e)
        print('Make sure to run this script from the repo root so `app` is importable.')
        return

    df = pd.read_csv(args.input)
    X_scaled, df_with_features = prepare_batch_inputs(df.copy(), scaler, feature_names)

    has_target = 'target' in df_with_features.columns
    y_true = df_with_features['target'].values if has_target else None

    metrics_rows = []
    for name, model in models.items():
        print('Evaluating', name)
        try:
            y_pred = model.predict(X_scaled)
        except Exception as e:
            print('Failed to predict with', name, e)
            continue

        row = {'Model': name}
        if has_target:
            row['Accuracy'] = accuracy_score(y_true, y_pred)
            row['Precision'] = precision_score(y_true, y_pred, zero_division=0)
            row['Recall'] = recall_score(y_true, y_pred, zero_division=0)
            row['F1-Score'] = f1_score(y_true, y_pred, zero_division=0)
        else:
            row['Accuracy'] = row['Precision'] = row['Recall'] = row['F1-Score'] = np.nan

        # ROC-AUC if probability or scores available
        roc_auc = None
        try:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_scaled)
                # if binary, take prob of positive class
                if probs.shape[1] == 2 and has_target:
                    y_score = probs[:, 1]
                    roc_auc = auc(*roc_curve(y_true, y_score)[:2])
            elif hasattr(model, 'decision_function'):
                y_score = model.decision_function(X_scaled)
                if has_target:
                    roc_auc = auc(*roc_curve(y_true, y_score)[:2])
        except Exception:
            roc_auc = None

        row['ROC-AUC'] = float(roc_auc) if roc_auc is not None else np.nan

        metrics_rows.append(row)

        # save confusion matrix if labeled
        safe_name = name.replace(' ', '_')
        if has_target:
            cm_path = os.path.join(args.out_dir, f'{safe_name}_confusion_matrix.png')
            plot_confusion_matrix(y_true, y_pred, labels=np.unique(y_true).tolist(), out_path=cm_path, title=f'Confusion Matrix: {name}')

        # ROC plot
        try:
            if has_target and 'y_score' in locals() or hasattr(model, 'predict_proba') or hasattr(model, 'decision_function'):
                # recompute y_score safely
                y_score = None
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_scaled)
                    if probs.shape[1] == 2:
                        y_score = probs[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_score = model.decision_function(X_scaled)

                if y_score is not None and has_target:
                    roc_path = os.path.join(args.out_dir, f'{safe_name}_roc_curve.png')
                    plot_roc(y_true, y_score, out_path=roc_path, title=f'ROC Curve: {name}')
        except Exception:
            pass

        # feature importances
        try:
            imp = None
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # flatten for binary / multiclass
                coef = model.coef_
                if coef.ndim == 1:
                    imp = coef
                else:
                    # for multiclass, sum absolute values across classes
                    imp = np.abs(coef).sum(axis=0)

            if imp is not None and feature_names is not None:
                fi_path = os.path.join(args.out_dir, f'{safe_name}_feature_importances.png')
                plot_feature_importances(feature_names, imp, out_path=fi_path, title=f'Feature Importances: {name}')
        except Exception:
            pass

    # write metrics table
    metrics_df = pd.DataFrame(metrics_rows).set_index('Model')
    metrics_csv = os.path.join(args.models_dir, 'model_comparison.csv')
    metrics_df.to_csv(metrics_csv)
    print('Wrote metrics to', metrics_csv)
    print('Wrote plots to', args.out_dir)


if __name__ == '__main__':
    main()

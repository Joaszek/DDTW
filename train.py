"""Main training script for DTW cost classification."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from config import config
from data_pipeline import load_all_data, split_train_val_test
from feature_engineering import get_feature_names


def get_models():
    """Define and return all models to train.

    Returns:
        Dictionary of model name to model instance
    """
    return {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000))
        ]),
        "SVM-RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale"))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=600,
            random_state=config.random_state
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            random_state=config.random_state
        )
    }


def plot_feature_importance(model, model_name, feature_names, top_n=20):
    """Plot and save feature importance for tree-based models.

    Args:
        model: Trained model instance
        model_name: Name of the model
        feature_names: List of feature names
        top_n: Number of top features to display
    """
    # Extract the actual model from pipeline if needed
    if hasattr(model, 'named_steps'):
        actual_model = model.named_steps.get('clf', model)
    else:
        actual_model = model

    # Get feature importance if available
    if hasattr(actual_model, 'feature_importances_'):
        importances = actual_model.feature_importances_

        # Create dataframe for easier handling
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        # Save to CSV
        importance_df.to_csv(f"results/{model_name}_feature_importance.csv", index=False)

        # Plot top N features
        top_features = importance_df.head(top_n)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'{model_name} - Top {top_n} Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"results/{model_name}_feature_importance.png", dpi=200, bbox_inches='tight')
        plt.close()

        print(f"\nTop 10 Most Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.6f}")


def train_and_evaluate_model(name, model, X_train, y_train, X_val, y_val,
                             X_test, y_test, categories, feature_names):
    """Train and evaluate a single model.

    Args:
        name: Model name
        model: Model instance
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        categories: List of category names
        feature_names: List of feature names
    """
    print("\n" + "=" * 80)
    print(f"MODEL: {name}")

    model.fit(X_train, y_train)

    # Evaluate on validation set
    pred_val = model.predict(X_val)
    acc_val = accuracy_score(y_val, pred_val)
    print(f"Validation Accuracy: {acc_val:.4f}")

    # Evaluate on test set
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, pred, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_test, pred, labels=categories)
    disp = ConfusionMatrixDisplay(cm, display_labels=categories)
    disp.plot(values_format="d")
    plt.title(f"{name} - Test Set")
    plt.savefig(f"results/{name}_confusion_matrix.png", dpi=200)
    plt.close()

    # Feature importance for tree-based models
    plot_feature_importance(model, name, feature_names)



def train_catboost(X_train, y_train, X_val, y_val, X_test, y_test,
                   categories, feature_names):
    """Train and evaluate CatBoost model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        categories: List of category names
        feature_names: List of feature names
    """
    try:
        from catboost import CatBoostClassifier

        cb = CatBoostClassifier(
            depth=8,
            learning_rate=0.05,
            iterations=1200,
            loss_function="MultiClass",
            eval_metric="Accuracy",
            random_seed=config.random_state,
            verbose=200
        )

        print("\n" + "=" * 80)
        print("MODEL: CatBoost")

        cb.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

        # Evaluate on validation set
        pred_val = cb.predict(X_val).reshape(-1)
        acc_val = accuracy_score(y_val, pred_val)
        print(f"Validation Accuracy: {acc_val:.4f}")

        # Evaluate on test set
        pred = cb.predict(X_test).reshape(-1)
        acc = accuracy_score(y_test, pred)
        print(f"Test Accuracy: {acc:.4f}")
        print("\nTest Set Classification Report:")
        print(classification_report(y_test, pred, digits=4))

        # Confusion matrix
        cm = confusion_matrix(y_test, pred, labels=categories)
        disp = ConfusionMatrixDisplay(cm, display_labels=categories)
        disp.plot(values_format="d")
        plt.title("CatBoost - Test Set")
        plt.savefig("results/CatBoost_confusion_matrix.png", dpi=200)
        plt.close()

        # Feature importance
        if hasattr(cb, 'feature_importances_'):
            importances = cb.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            importance_df.to_csv("results/CatBoost_feature_importance.csv", index=False)

            # Plot top 20 features
            top_features = importance_df.head(20)
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('CatBoost - Top 20 Features')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig("results/CatBoost_feature_importance.png", dpi=200, bbox_inches='tight')
            plt.close()

            print(f"\nTop 10 Most Important Features:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"  {row['feature']:30s}: {row['importance']:.6f}")

    except Exception as e:
        print("\n[INFO] CatBoost skipped (library missing or import error).")
        print("If you want CatBoost: pip install catboost")
        print(f"Error: {repr(e)}")


def main():
    """Main training pipeline."""
    # Load data - using only BAS and B6 categories as in the original code
    categories_to_use = ["BAS", "B6"]
    X, y, meta, all_rows = load_all_data(categories=categories_to_use)

    # Get feature names
    feature_names = get_feature_names()
    print(f"\nTotal features: {len(feature_names)}")

    # Split train/val/test with time-awareness to prevent leakage
    print("\nSplitting data into train/val/test...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(all_rows, X, y)

    # Train and evaluate standard models
    models = get_models()
    for name, model in models.items():
        train_and_evaluate_model(
            name, model, X_train, y_train, X_val, y_val,
            X_test, y_test, categories_to_use, feature_names
        )

    # Train CatBoost (optional)
    train_catboost(X_train, y_train, X_val, y_val, X_test, y_test,
                   categories_to_use, feature_names)


if __name__ == "__main__":
    main()

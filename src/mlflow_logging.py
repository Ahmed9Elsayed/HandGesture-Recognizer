import xgboost as xgb
import numpy as np
import mlflow
import mlflow.data
import mlflow.models
import mlflow.sklearn
from pathlib import Path
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=6)
    param_grid = {
        'n_estimators': [100, 200, 400],
        'max_depth': [10, 20, 30],
        'criterion': ["gini", "entropy"],
        'max_leaf_nodes': [50, 100]
    }
    grid = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, scoring='f1_weighted', verbose=0)
    model = grid.fit(X_train, y_train)
    return model

def train_SVM(X_train, y_train):
    svm = SVC(random_state=6)
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ["rbf", "poly"],
        'class_weight': [None, 'balanced']
    }
    grid = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1, scoring='f1_weighted', verbose=0)
    model = grid.fit(X_train, y_train)
    return model

def train_XGBOOST(X_train, y_train):
    num_classes = len(np.unique(y_train))
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01],
        'n_estimators': [100, 200],
        'average': ['weighted']
    }

    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob', 
        num_class=num_classes,
        eval_metric='mlogloss',
        use_label_encoder=False  
    )

    grid_search = GridSearchCV(
        estimator=xgb_model, 
        param_grid=param_grid, 
        cv=3, 
        scoring='f1_weighted', 
        verbose=1
    )
    model = grid_search.fit(X_train, y_train)
    return model

def setup_mlflow_experiment(experiment_name: str, tracking_uri: str = "http://127.0.0.1:5000/") -> str:
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment(experiment_name)
    return exp.experiment_id

def eval_metrics(actual, pred):
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, average='weighted')
    prec = metrics.precision_score(actual, pred, average='weighted')
    recall = metrics.recall_score(actual, pred, average='weighted')
    return accuracy, f1, prec, recall

def log_model_with_mlflow(model, chosen_params, X_test, y_test, model_name: str, exp_id: str, output_dir: Path):
    with mlflow.start_run(experiment_id=exp_id, run_name=model_name) as run:

        mlflow.set_tag("model", model_name)

        pred = model.predict(X_test)
        accuracy, f1, prec, recall = eval_metrics(y_test, pred)

        mlflow.log_params(chosen_params)
        mlflow.log_metrics({
            "Accuracy": accuracy,
            "f1-score": f1,
            'Recall': recall,
            'Precision': prec
        })

        mlflow.log_artifact(str(output_dir / "confusion.png"))

        pd_dataset = mlflow.data.from_pandas(X_test, name="Testing Dataset")
        mlflow.log_input(pd_dataset, context="Testing")

        signature = mlflow.models.infer_signature(X_test, y_test)
        mlflow.sklearn.log_model(model, model_name, signature=signature, input_example=X_test.iloc[[0]])
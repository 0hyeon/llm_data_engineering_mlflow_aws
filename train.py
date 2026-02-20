import os
import tempfile
import mlflow
import numpy as np
from data import X_train, X_val, y_train, y_val
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import ParameterGrid
from params import elasticnet_param_grid
from utils import eval_metrics

for params in ParameterGrid(elasticnet_param_grid):
    with mlflow.start_run():
        lr = ElasticNet(**params)
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_val)
        metrics = eval_metrics(y_val, y_pred)

        mlflow.log_input(
            mlflow.data.from_numpy(X_train.toarray()), context="Training data"
        )
        mlflow.log_input(
            mlflow.data.from_numpy(X_val.toarray()), context="Validation data"
        )

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # ✅ (기존) Logged Model 저장: S3에 1/models/m-... 생성됨 (유지)
        mlflow.sklearn.log_model(
            lr,
            "ElasticNet model",
            input_example=X_train,
            code_paths=["train.py", "data.py", "params.py", "utils.py"],
        )

        # ✅ (추가) Run Artifacts로도 저장: S3에 1/<run_id>/artifacts/... 생성
        # 1) 코드 파일을 run artifacts로 저장 (Artifacts 탭에서 보기 좋음)
        for p in ["train.py", "data.py", "params.py", "utils.py"]:
            mlflow.log_artifact(p, artifact_path="code")

        # 2) 모델을 run artifacts로 저장 (run_id/artifacts/model/...)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "model")
            mlflow.sklearn.save_model(lr, path=model_dir)
            mlflow.log_artifacts(model_dir, artifact_path="model")

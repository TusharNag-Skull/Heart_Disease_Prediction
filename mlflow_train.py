import mlflow
import mlflow.sklearn
import joblib

# pointing to MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Heart_Disease_Experiment")

# Loading the  model which will be used for the actual final prediction
model = joblib.load("final_lgbm.pkl")

with mlflow.start_run():
    mlflow.log_param("model_type", "LightGBM")

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",  # artifact_path is deprecated
        registered_model_name="HeartDiseaseModel"
    )

print("Model logged and registered successfully")
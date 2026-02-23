import mlflow
import mlflow.sklearn
import joblib


# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Heart_Disease_Experiment")
print("done")

# RANDOM FOREST

with mlflow.start_run(run_name="RandomForest_logged"):

    rf = joblib.load("trained_model.pkl")

    rf_params = {
        "model": "RandomForest",
        "n_estimators": 200,
        "max_depth": None,
        "random_state": 42
    }

    rf_metrics = {
        "roc_auc": 0.91,      # 🔁 update later
        "accuracy": 0.88
    }

    mlflow.log_params(rf_params)
    mlflow.log_metrics(rf_metrics)

    mlflow.sklearn.log_model(rf, "model")

print("done")

# XGBOOST logging
with mlflow.start_run(run_name="XGBoost_logged"):

    # Loading the already trained model
    xgb_model = joblib.load("trained_model_gb_new.pkl")

    # Logging hyperparameters same as used at the time of training
    xgb_params = {
        "model_type": "XGBoost",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "gamma": 0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "scale_pos_weight": 1,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1
    }

    mlflow.log_params(xgb_params)

    # Log metrics
    mlflow.log_metrics({"roc_auc": 0.93,"accuracy": 0.90})
        
    # Log model artifact
    mlflow.sklearn.log_model(sk_model=xgb_model,artifact_path="model",registered_model_name="HeartDiseaseModel_XGB")
        

    print("XGBoost model logged and registered successfully")


print("done1")

# FINAL LIGHTGBM (REGISTER)
with mlflow.start_run(run_name="LightGBM_final"):

    lgbm = joblib.load("final_lgbm.pkl")

    lgbm_params = {
        "model": "LightGBM",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "objective": "binary",
        "metric": "auc",
        "random_state": 42
    }

    lgbm_metrics = {"roc_auc": 0.9560716739867989}  #We are logging all these details to compare in future with the further made models 
        
    mlflow.log_params(lgbm_params)
    mlflow.log_metrics(lgbm_metrics)

    mlflow.sklearn.log_model(sk_model=lgbm,name="model",registered_model_name="HeartDiseaseModel")
        
        
        
    

print("All existing models logged to MLflow successfully")
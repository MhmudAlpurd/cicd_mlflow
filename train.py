import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load California Housing Dataset
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set the MLflow experiment name
mlflow.set_experiment("California_House_Price_Estimator")

# Start an MLflow run
with mlflow.start_run():
    # Define hyperparameters
    n_estimators = 100
    max_depth = 10
    random_state = 42
    
    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)
    
    # Train model
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Evaluate model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    # Log metrics
    mlflow.log_metric("mse", mse)
    
    # Log the model
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    print(f"Logged model with MSE: {mse}")

# Instructions to view logs:
# Run `mlflow ui` in the terminal and open http://localhost:5000

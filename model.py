import xgboost as xgb
import joblib
from sklearn.metrics import classification_report
from preprocess import preprocess_input

def train_and_save_model():
    # Get preprocessed data - no fit parameter needed now
    X_train, X_test, y_train, y_test = preprocess_input()  # Removed fit=True
    
    # Train XGBoost model
    model = xgb.XGBClassifier(
        eval_metric="logloss",
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "mental_health_model.pkl")
    print("✅ Model saved as mental_health_model.pkl")

    # Evaluate
    y_pred = model.predict(X_test)
    print("\n📊 Model Evaluation Report:\n")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_and_save_model()
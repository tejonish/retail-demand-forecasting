#Inference Pipeline
import joblib
import pandas as pd

# Load model and features
model = joblib.load("models/xgb_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

def predict(input_df):
    # Align columns
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(input_df)

    return prediction

if __name__ == "__main__":
    print("Inference pipeline working ✅")
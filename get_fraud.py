import pandas as pd
import joblib
import json

csv_path = "c:/Users/PRIYANSHU SINGHA ROY/OneDrive/Desktop/Fintellix/Fintellix-ML-Model/data/creditcard.csv"
scaler_path = "c:/Users/PRIYANSHU SINGHA ROY/OneDrive/Desktop/Fintellix/Fintellix-ML-Model/models/amount_scaler.pkl"

df = pd.read_csv(csv_path)
fraud_rows = df[df["Class"] == 1]
if len(fraud_rows) > 0:
    first_fraud = fraud_rows.iloc[0].copy()
    print("Found Fraud Row ID:", first_fraud.get("id", "Unknown"))
    
    scaler = joblib.load(scaler_path)
    
    # Scale Amount and Time based on how scaler was fitted
    scaled_vals = scaler.transform([[first_fraud["Amount"], first_fraud["Time"]]])
    scaled_amt, scaled_time = scaled_vals[0]
    
    features = [
        scaled_time,
        first_fraud["V1"], first_fraud["V2"], first_fraud["V3"], first_fraud["V4"],
        first_fraud["V5"], first_fraud["V6"], first_fraud["V7"], first_fraud["V8"],
        first_fraud["V9"], first_fraud["V10"], first_fraud["V11"], first_fraud["V12"],
        first_fraud["V13"], first_fraud["V14"], first_fraud["V15"], first_fraud["V16"],
        first_fraud["V17"], first_fraud["V18"], first_fraud["V19"], first_fraud["V20"],
        first_fraud["V21"], first_fraud["V22"], first_fraud["V23"], first_fraud["V24"],
        first_fraud["V25"], first_fraud["V26"], first_fraud["V27"], first_fraud["V28"],
        scaled_amt
    ]
    
    print("FEATURES:")
    print(json.dumps(features))
    
    # Try all amount combinations
    print("Target Pattern:")
    print("V1-V28:", list(features[1:29]))
    print("Scaled Time:", scaled_time)
    print("Scaled Amount:", scaled_amt)
else:
    print("No fraud rows found in CSV!")

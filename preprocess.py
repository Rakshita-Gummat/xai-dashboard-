import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def preprocess_input(csv_path='survey.csv', single_input=None):
    """
    Process data from CSV (batch mode) or single input (prediction mode)
    """
    # Initialize encoders and scaler
    encoders = {}
    scaler = StandardScaler()
    
    if single_input is None:
        # Batch processing mode (training)
        df = pd.read_csv(csv_path)
        
        # Original preprocessing steps
        df.drop(['comments', 'state', 'Timestamp'], axis=1, inplace=True)
        df['treatment'] = df['treatment'].map({'Yes': 1, 'No': 0})
        
        # Fill missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna("Unknown")
            else:
                df[col] = df[col].fillna(df[col].mean())
        
        # Clean column names
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
        
        # Feature selection
        keep_columns = ['age', 'gender', 'country', 'self_employed', 'family_history',
                      'remote_work', 'benefits', 'care_options', 'wellness_program',
                      'seek_help', 'anonymity', 'leave', 'mental_health_consequence',
                      'coworkers', 'supervisor', 'mental_health_interview',
                      'phys_health_interview', 'mental_vs_physical',
                      'obs_consequence', 'treatment']
        df = df[keep_columns]
        
        # Age bounds
        df = df[(df['age'] > 10) & (df['age'] < 100)]
        
        # Encode categoricals and save encoders
        for col in df.select_dtypes(include='object').columns:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col])
        joblib.dump(encoders, 'encoders.joblib')
        
        # Scale numerical
        df['age'] = scaler.fit_transform(df[['age']])
        joblib.dump(scaler, 'scaler.joblib')
        
        # Split data
        X = df.drop('treatment', axis=1)
        y = df['treatment']
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    else:
        # Single prediction mode
        df = pd.DataFrame([single_input])
        
        # Clean column names to match training
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
        
        # Ensure all expected columns are present
        expected_cols = [col for col in keep_columns if col != 'treatment']
        for col in expected_cols:
            if col not in df.columns:
                df[col] = "Unknown" if pd.api.types.is_string_dtype(df[col]) else 0
        
        # Apply saved transformations
        encoders = joblib.load('encoders.joblib')
        for col in encoders:
            try:
                df[col] = encoders[col].transform(df[col])
            except ValueError:
                df[col] = encoders[col].transform(["Unknown"])[0]
        
        # Scale age
        scaler = joblib.load('scaler.joblib')
        df['age'] = scaler.transform(df[['age']])
        
        return df[expected_cols]
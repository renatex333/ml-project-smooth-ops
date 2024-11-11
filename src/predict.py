import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib  
from process import load_data  

def predict(file_path):
    model = joblib.load("../models/model.pkl")  
    
    data = load_data("../data/winequality_predict.csv")
    
    X_to_predict = data.drop(columns=['quality'])  
    
    X = X_to_predict  
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  

    predictions = model.predict(X_scaled)
    
    results = pd.DataFrame({
        "Predicted Quality": predictions
    })
    
    results.to_csv("predictions.csv", index=False)
    print("Predictions saved to 'predictions.csv'.")

if __name__ == "__main__":
    
    predict("../data/winequality_predict.csv")

import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import joblib  # Add this for saving/loading models
from process import load_data  

def train_model():
    data = load_data("../data/winequality-red.csv")
    
    X = data.drop('quality', axis=1)
    y = data['quality']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    models = {
        'SVC': SVC(),
        'SGDClassifier': SGDClassifier(),
        'RandomForestClassifier': RandomForestClassifier()
    }
    
    best_model = None
    best_score = 0
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        score = model.score(X_test, y_test)
        if score > best_score:
            best_model = model
            best_score = score
        
        print(f"{model_name} Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))
    
    # Save the best-performing model to 'model.pkl'
    if best_model is not None:
        joblib.dump(best_model, 'model.pkl')
        print(f"\nBest model ({type(best_model).__name__}) saved as 'model.pkl'.")
    
    # Optional: GridSearchCV for hyperparameter tuning (example with SVC)
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
    grid = GridSearchCV(SVC(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("\nBest parameters for SVC:", grid.best_params_)
    print("Best cross-validation score for SVC:", grid.best_score_)

    # Save the GridSearchCV best estimator if desired
    joblib.dump(grid.best_estimator_, 'best_model_gridsearch.pkl')
    print("Best GridSearchCV model saved as 'best_model_gridsearch.pkl'.")

if __name__ == "__main__":
    train_model()


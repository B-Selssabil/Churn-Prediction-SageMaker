
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
import joblib
import boto3
from io import StringIO
import argparse
import os
import numpy as np
import pandas as pd

# loading the model
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__ == "__main__":
    
    print("['INFO'] Extracting arguments")
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)
    
    # Data, model and output directories
    
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
  
    parser.add_argument('--train-file', type=str, default="Churn-train-V-1.csv")
    parser.add_argument('--test-file', type=str, default="Churn-test-V-1.csv")

    args, _ = parser.parse_known_args()
    
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))
    
    features = list(train_df)
    features.remove('Churn_Yes')
    target = "Churn_Yes"
    
    print("Building training and testin datasets")
    print()

    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[target]
    y_test = test_df[target]
    
    print("Columns :")
    print(features)
    print()

    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state)
    model.fit(X_train, y_train)
    print()

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print('Model presisted at '+ model_path)
    print()
    
    
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_rep = classification_report(y_test, y_pred_test)
    
    print()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import joblib
import os
import json

def train_model():
    print("Loading dataset...")
    local_path = "data/twitter_sentiment_dataset.csv"
    url = "https://raw.githubusercontent.com/zfz/twitter_corpus/master/full-corpus.csv"
    
    try:
        if os.path.exists(local_path):
            df = pd.read_csv(local_path)
        else:
            df = pd.read_csv(url)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    df = df[['Sentiment', 'TweetText']]
    df = df[df['Sentiment'].isin(['positive', 'negative', 'neutral'])]
    
    # Balance dataset
    min_size = min(df['Sentiment'].value_counts())
    df = pd.concat([
        df[df['Sentiment'] == s].sample(min_size, random_state=42) 
        for s in df['Sentiment'].unique()
    ]).sample(frac=1, random_state=42)

    X = df['TweetText']
    y = df['Sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Vectorizing...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Models to compare
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(dual=False)) # Calibrated for predict_proba
    }

    results = {}
    best_acc = 0
    best_model_name = ""

    print("Training and evaluating models...")
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        results[name] = round(acc * 100, 2)
        print(f"{name} Accuracy: {acc:.2%}")
        
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            # Save the best model to the centralized models/ directory
            joblib.dump(model, 'models/model.pkl')

    # Save vectorizer
    joblib.dump(vectorizer, 'models/vectorizer.pkl')

    # Save comparison results for the dashboard
    with open('models/model_comparison.json', 'w') as f:
        json.dump({
            "results": results,
            "best_model": best_model_name,
            "best_accuracy": round(best_acc * 100, 2)
        }, f)

    print(f"\nTraining Complete! Best model: {best_model_name} ({best_acc:.2%})")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    train_model()

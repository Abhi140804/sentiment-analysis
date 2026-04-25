import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_model():
    print("Loading dataset...")
    local_path = "data/twitter_sentiment_dataset.csv"
    url = "https://raw.githubusercontent.com/zfz/twitter_corpus/master/full-corpus.csv"
    
    try:
        if os.path.exists(local_path):
            print(f"Using local dataset: {local_path}")
            df = pd.read_csv(local_path)
        else:
            print(f"Downloading dataset from: {url}")
            df = pd.read_csv(url)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Filter for relevant columns and remove 'irrelevant' class
    df = df[['Sentiment', 'TweetText']]
    df = df[df['Sentiment'].isin(['positive', 'negative', 'neutral'])]
    
    # Balance the dataset (Downsample Neutral to be closer to others)
    neutral_df = df[df['Sentiment'] == 'neutral']
    positive_df = df[df['Sentiment'] == 'positive']
    negative_df = df[df['Sentiment'] == 'negative']
    
    min_size = min(len(positive_df), len(negative_df), len(neutral_df))
    neutral_df = neutral_df.sample(min_size, random_state=42)
    positive_df = positive_df.sample(min_size, random_state=42)
    negative_df = negative_df.sample(min_size, random_state=42)
    
    df = pd.concat([positive_df, negative_df, neutral_df]).sample(frac=1, random_state=42)
    
    print(f"Dataset shape after filtering: {df.shape}")
    print(f"Class distribution:\n{df['Sentiment'].value_counts()}")

    # Preprocessing
    X = df['TweetText']
    y = df['Sentiment']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorization
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Model Training
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
    model.fit(X_train_tfidf, y_train)

    # Evaluation
    y_pred = model.predict(X_test_tfidf)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Saving artifacts
    print("Saving model and vectorizer...")
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("Done!")

if __name__ == "__main__":
    train_model()

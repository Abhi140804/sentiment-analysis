import joblib
import os
import streamlit as st

class SentimentPredictor:
    def __init__(self, model_path='models/model.pkl', vectorizer_path='models/vectorizer.pkl'):
        """
        Initialize the predictor by loading the pre-trained model and vectorizer.
        """
        # Adjusted paths for new modular structure
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        m_path = os.path.join(base_dir, model_path)
        v_path = os.path.join(base_dir, vectorizer_path)
        
        if not os.path.exists(m_path) or not os.path.exists(v_path):
            raise FileNotFoundError(f"Model or Vectorizer not found at {m_path}. Run training first.")
        
        self.model = joblib.load(m_path)
        self.vectorizer = joblib.load(v_path)

    def predict(self, text):
        """
        Predict the sentiment of the given text.
        Returns: sentiment (str), confidence (float)
        """
        if not text.strip():
            return "Neutral", 0.0
        
        text_vectorized = self.vectorizer.transform([text])
        prediction = self.model.predict(text_vectorized)[0]
        probabilities = self.model.predict_proba(text_vectorized)[0]
        
        class_idx = list(self.model.classes_).index(prediction)
        confidence = probabilities[class_idx]
        
        return prediction.capitalize(), float(confidence)

    def predict_batch(self, texts):
        """
        Predict sentiment for a list of texts efficiently.
        Returns: list of (sentiment, confidence)
        """
        if not texts:
            return []
            
        # Vectorize all texts at once
        vectors = self.vectorizer.transform(texts)
        
        # Batch prediction
        predictions = self.model.predict(vectors)
        probabilities = self.model.predict_proba(vectors)
        classes = list(self.model.classes_)
        
        results = []
        for i, pred in enumerate(predictions):
            conf = probabilities[i][classes.index(pred)]
            results.append((pred.capitalize(), float(conf)))
        return results

@st.cache_resource
def get_predictor():
    """
    Cached function to load the model once.
    """
    return SentimentPredictor()

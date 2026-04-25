# AI-Powered Real-Time Sentiment Analytics System 📊🐦

A professional-grade Natural Language Processing (NLP) dashboard designed for real-time sentiment analysis of text and live social media streams. This project is built using Python, Scikit-Learn, and Streamlit, featuring a modular architecture and an interactive analytics suite.

## 🚀 Key Features

- **Real-Time Text Analysis**: Instantly classify any text input as Positive, Negative, or Neutral.
- **Twitter (X) Live Tracker**: Fetch and analyze live tweets based on keywords using X API v2.
- **Advanced Visual Dashboard**:
  - **Sentiment Distribution**: Dynamic pie and bar charts.
  - **Trend Analysis**: Line graphs tracking sentiment momentum over time.
  - **Linguistic Insights**: Separate Word Clouds for Positive and Negative sentiments.
  - **Reliability Metrics**: Confidence distribution histograms.
- **High-Performance Engine**: Optimized with batch processing and model caching for seamless updates.
- **Enterprise UI**: A sleek, dark-themed dashboard with top-tab navigation and responsive design.

## 🏗️ Architecture

The project follows a clean, modular design suitable for final-year engineering showcases:

```text
├── analytics/         # Visualization logic (Plotly/Matplotlib)
├── app/               # Streamlit UI implementation
├── core/              # ML Prediction, Twitter API, & Model logic
├── data/              # Local dataset storage (CSV)
├── models/            # Serialized ML model and TF-IDF vectorizer
└── requirements.txt   # Project dependencies
```

## 🛠️ Tech Stack

- **ML Engine**: Logistic Regression (Multinomial)
- **Vectorization**: TF-IDF with N-gram (1, 2) support
- **Visualization**: Plotly, WordCloud, Matplotlib
- **API Integration**: Tweepy (X API v2)
- **Frontend**: Streamlit
- **Data Handling**: Pandas, NumPy

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Abhi140804/sentiment-analysis.git
cd sentiment-analysis
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Train Model
If you want to re-train the model with the local dataset:
```bash
python3 core/model.py
```

### 5. Run the Application
```bash
streamlit run app/app.py
```

## 🐦 Twitter API Configuration

To use the live tracking feature, you will need an **X API Bearer Token**:
1. Get a token from the [X Developer Portal](https://developer.x.com/).
2. Paste the token into the **Twitter API Configuration** expander within the app's Twitter Tracker tab.

## 📈 Dashboard Insights

- **Pie Chart**: Visualizes the overall sentiment mix.
- **Trend Graph**: Tracks how sentiment scores evolve during your session.
- **Word Clouds**: Uncover specific keywords driving positive vs. negative sentiments.

## 👥 Authors
- **Abhi** - *Project Lead & Developer*

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

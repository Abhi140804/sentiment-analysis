import streamlit as st
import pandas as pd
import sys
import os
import json
from datetime import datetime
# Professional Analytics Dashboard v3.8 - Multi-Model Edition (Cache Cleared)
import time

# Add root to sys.path to allow modular imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.predict import get_predictor
from core.twitter_fetch import TwitterFetcher
import visuals.engine as vis

# Page configuration
st.set_page_config(
    page_title="Sentiment Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom Styling for SaaS Look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .stApp { background-color: #020617; color: #f8fafc; }
    
    .dashboard-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    
    .main-title {
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(135deg, #60a5fa 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .section-header {
        font-size: 1rem;
        font-weight: 700;
        color: #3b82f6;
        text-transform: uppercase;
        letter-spacing: 0.15rem;
        margin-bottom: 1.5rem;
        border-left: 3px solid #3b82f6;
        padding-left: 1rem;
    }
    
    /* Result styling */
    .metric-card {
        text-align: center;
        padding: 1.2rem;
        border-radius: 16px;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255,255,255,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Session State
if 'history' not in st.session_state:
    st.session_state.history = []
if 'twitter_data' not in st.session_state:
    st.session_state.twitter_data = None

# Resource Loading
predictor = get_predictor()

# --- TOP NAVIGATION BAR ---
col_logo, col_info = st.columns([1, 1])
with col_logo:
    st.markdown('<h1 class="main-title">Sentiment AI Intelligence</h1>', unsafe_allow_html=True)
with col_info:
    st.markdown('<p style="text-align: right; color: #64748b; margin-top: 2rem;">Professional Analytics Dashboard v3.8</p>', unsafe_allow_html=True)

st.markdown("---")

# TOP MENU
mode = st.tabs(["📊 Live Dashboard", "🧠 Model Comparison", "⚙️ System Info"])

# --- TAB 1: LIVE DASHBOARD ---
with mode[0]:
    # Main Dashboard Layout
    col_left, col_right = st.columns([1.2, 2], gap="large")

    # --- LEFT PANEL: INPUT ---
    with col_left:
        st.markdown('<p class="section-header">📡 Data Ingestion</p>', unsafe_allow_html=True)
        input_mode = st.radio("Choose Input Source:", ["Manual Text Analysis", "X (Twitter) Live Tracker"], horizontal=True)
        
        if input_mode == "Manual Text Analysis":
            st.markdown("#### 🖊️ Text Input")
            text_in = st.text_area("Type your text here:", placeholder="e.g. I am really impressed!", height=150, label_visibility="collapsed")
            if st.button("🚀 Analyze Text", use_container_width=True):
                if text_in.strip():
                    with st.spinner("Classifying..."):
                        sent, conf = predictor.predict(text_in)
                        st.session_state.history.append({
                            "Timestamp": datetime.now(),
                            "Source": "Manual",
                            "Text": text_in,
                            "Sentiment": sent,
                            "Confidence_Val": conf,
                            "Confidence": f"{conf:.1%}"
                        })
                        st.success(f"Classification Complete: **{sent}** ({conf:.1%})")
                        time.sleep(0.5)
                        st.rerun()
        else:
            st.markdown("#### 🐦 Twitter Tracker Config")
            token = st.text_input("X API Bearer Token", type="password")
            keyword = st.text_input("Search Keyword", placeholder="e.g. #AI, Apple")
            limit = st.slider("Samples", 10, 100, 20)
            
            if st.button("📡 Fetch Live Stream", use_container_width=True):
                if not token or not keyword:
                    st.error("Token and Keyword required.")
                else:
                    try:
                        with st.spinner(f"Connecting to X for '{keyword}'..."):
                            fetcher = TwitterFetcher(token)
                            tweets = fetcher.fetch_tweets(keyword, max_results=limit)
                            if tweets:
                                results = predictor.predict_batch(tweets)
                                timestamp = datetime.now()
                                for t, (s, c) in zip(tweets, results):
                                    st.session_state.history.append({
                                        "Timestamp": timestamp,
                                        "Source": f"Twitter: {keyword}",
                                        "Text": t, "Sentiment": s, "Confidence_Val": c, "Confidence": f"{c:.1%}"
                                    })
                                st.session_state.twitter_data = {"keyword": keyword, "tweets": tweets}
                                st.rerun()
                    except Exception as e:
                        st.error(f"Fetch failed: {e}")

            if st.button("🎭 Simulate Live Data (No API Required)", use_container_width=True):
                if not keyword: st.error("Enter keyword first.")
                else:
                    with st.spinner(f"Simulating '{keyword}'..."):
                        time.sleep(1.2)
                        mock_tweets = [f"I love {keyword}!", f"Just tried {keyword}, not bad.", f"{keyword} is amazing!", f"Issues with {keyword} today.", f"Future of {keyword} is bright!"]
                        results = predictor.predict_batch(mock_tweets)
                        for t, (s, c) in zip(mock_tweets, results):
                            st.session_state.history.append({"Timestamp": datetime.now(), "Source": f"Simulated: {keyword}", "Text": t, "Sentiment": s, "Confidence_Val": c, "Confidence": f"{c:.1%}"})
                        st.session_state.twitter_data = {"keyword": keyword, "tweets": mock_tweets}
                        st.rerun()

        if st.session_state.history:
            df_sum = pd.DataFrame(st.session_state.history)
            st.markdown('<p class="section-header">📉 Session Summary</p>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total", len(df_sum))
            c2.metric("Pos%", f"{(len(df_sum[df_sum['Sentiment']=='Positive'])/len(df_sum)):.0%}")
            c3.metric("Avg Conf", f"{df_sum['Confidence_Val'].mean():.1%}")

    # --- RIGHT PANEL: ANALYTICS ---
    with col_right:
        st.markdown('<p class="section-header">📊 Visual Intelligence</p>', unsafe_allow_html=True)
        if not st.session_state.history:
            st.info("Ingest data on the left to activate analytics.")
        else:
            df_viz = pd.DataFrame(st.session_state.history)
            r1c1, r1c2 = st.columns(2)
            with r1c1: st.plotly_chart(vis.create_sentiment_pie(df_viz), use_container_width=True)
            with r1c2: st.plotly_chart(vis.create_sentiment_bar(df_viz), use_container_width=True)
            
            r2c1, r2c2 = st.columns(2)
            with r2c1: 
                t_fig = vis.create_trend_graph(df_viz)
                if t_fig: st.plotly_chart(t_fig, use_container_width=True)
            with r2c2: 
                c_fig = vis.create_confidence_dist(df_viz)
                if c_fig: st.plotly_chart(c_fig, use_container_width=True)

            st.markdown("### ☁️ Word Clouds")
            cp, cn = st.columns(2)
            with cp:
                st.markdown("<p style='text-align:center; color:#10b981;'>Positive</p>", unsafe_allow_html=True)
                fp = vis.generate_wordcloud(" ".join(df_viz[df_viz['Sentiment'] == 'Positive']['Text']))
                if fp: st.pyplot(fp)
            with cn:
                st.markdown("<p style='text-align:center; color:#ef4444;'>Negative</p>", unsafe_allow_html=True)
                fn = vis.generate_wordcloud(" ".join(df_viz[df_viz['Sentiment'] == 'Negative']['Text']), color='blue')
                if fn: st.pyplot(fn)

# --- TAB 2: MODEL COMPARISON ---
with mode[1]:
    st.markdown('<p class="section-header">🧠 Model Performance Dashboard</p>', unsafe_allow_html=True)
    
    comp_path = 'models/model_comparison.json'
    if os.path.exists(comp_path):
        with open(comp_path, 'r') as f:
            comp_data = json.load(f)
        
        col_m_left, col_m_right = st.columns([1.5, 1])
        
        with col_m_left:
            st.plotly_chart(vis.create_model_comparison_chart(comp_data), use_container_width=True)
            
        with col_m_right:
            st.markdown("#### 🏆 Best Model")
            st.markdown(f"""
                <div class="metric-card">
                    <p style="color: #ffcc00; font-size: 1.2rem; font-weight: 800;">{comp_data['best_model']}</p>
                    <p style="font-size: 2.5rem; font-weight: 800;">{comp_data['best_accuracy']}%</p>
                    <p style="color: #64748b; font-size: 0.8rem;">Current Production Model</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📋 Performance Table")
            df_perf = pd.DataFrame(list(comp_data['results'].items()), columns=['Algorithm', 'Accuracy (%)'])
            st.table(df_perf.sort_values('Accuracy (%)', ascending=False))
            
        st.info("💡 **Note:** All models are trained on the same balanced dataset using TF-IDF vectorization with bigram support.")
    else:
        st.warning("Model comparison data not found. Please run the training module to generate metrics.")

# --- TAB 3: SYSTEM INFO & LOGS ---
with mode[2]:
    st.markdown('<p class="section-header">⚙️ System Info & History</p>', unsafe_allow_html=True)
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history).sort_values('Timestamp', ascending=False), use_container_width=True, hide_index=True)
        if st.button("🗑️ Reset All Session Data", use_container_width=True):
            st.session_state.history = []
            st.session_state.twitter_data = None
            st.rerun()
    else:
        st.info("No activity logs available for this session.")

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center; color: #334155; font-size: 0.8rem; margin: 2rem 0;">AI SENTIMENT ANALYTICS PRO • FINAL YEAR PROJECT PORTFOLIO • &copy; 2026</div>', unsafe_allow_html=True)

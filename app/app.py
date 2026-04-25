import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime
import time

# Add root to sys.path to allow modular imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.predict import get_predictor
from core.twitter_fetch import TwitterFetcher
from analytics.visualize import (
    create_sentiment_pie, create_sentiment_bar, create_trend_graph, 
    create_confidence_dist, generate_wordcloud, create_hashtag_chart
)

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
    .stApp { background-color: #0f172a; color: #f8fafc; }
    
    .dashboard-card {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .main-title {
        font-weight: 800;
        font-size: 2.5rem;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid #38bdf8;
        padding-left: 1rem;
    }
    
    /* Result styling */
    .metric-card {
        text-align: center;
        padding: 1rem;
        border-radius: 12px;
        background: rgba(255,255,255,0.03);
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
    st.markdown('<p style="text-align: right; color: #64748b; margin-top: 2rem;">Professional Analytics Dashboard v3.5</p>', unsafe_allow_html=True)

st.markdown("---")

# --- MAIN DASHBOARD LAYOUT ---
# Left: Input (1.2) | Right: Analytics (2)
col_left, col_right = st.columns([1.2, 2], gap="large")

# --- LEFT PANEL: INPUT ---
with col_left:
    st.markdown('<p class="section-header">📡 Data Ingestion</p>', unsafe_allow_html=True)
    
    input_mode = st.radio("Choose Input Source:", ["Manual Text Analysis", "X (Twitter) Live Tracker"], horizontal=True)
    
    with st.container():
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        
        if input_mode == "Manual Text Analysis":
            st.markdown("#### 🖊️ Text Input")
            text_in = st.text_area("Type your text here:", placeholder="e.g. I am really impressed with this system!", height=150, label_visibility="collapsed")
            if st.button("🚀 Analyze Text", use_container_width=True):
                if text_in.strip():
                    with st.spinner("Classifying..."):
                        sent, conf = predictor.predict(text_in)
                        # Add to history
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
            keyword = st.text_input("Search Keyword", placeholder="e.g. #AI, Apple, Tesla")
            limit = st.slider("Samples", 10, 100, 20)
            
            if st.button("📡 Fetch Live Stream", use_container_width=True):
                if not token or not keyword:
                    st.error("Token and Keyword are required.")
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
                                        "Text": t,
                                        "Sentiment": s,
                                        "Confidence_Val": c,
                                        "Confidence": f"{c:.1%}"
                                    })
                                st.session_state.twitter_data = {"keyword": keyword, "tweets": tweets}
                                st.rerun()
                    except Exception as e:
                        st.error(f"Fetch failed: {e}")
                        st.info("💡 **Pro-Tip:** Your API Tier might be restricted. Use the 'Simulate' button below to test the dashboard features!")

            if st.button("🎭 Simulate Live Data (No API Required)", use_container_width=True):
                if not keyword:
                    st.error("Please enter a keyword first.")
                else:
                    with st.spinner(f"Simulating live stream for '{keyword}'..."):
                        time.sleep(1.5)
                        mock_tweets = [
                            f"I love how {keyword} is changing the industry! #innovation",
                            f"Just tried {keyword} and it was a bit disappointing. #feedback",
                            f"{keyword} is absolutely amazing, best experience ever! #happy",
                            f"Not sure about {keyword}, still has some bugs to fix. #neutral",
                            f"The new update for {keyword} is a game changer! 🔥",
                            f"Is anyone else having issues with {keyword} today? #help",
                            f"Great session today learning about {keyword}! #learning",
                            f"I hate how expensive {keyword} has become. #sad",
                            f"So excited for the future of {keyword}! 🚀",
                            f"{keyword} is just okay, nothing special. #honestreview"
                        ]
                        results = predictor.predict_batch(mock_tweets)
                        timestamp = datetime.now()
                        for t, (s, c) in zip(mock_tweets, results):
                            st.session_state.history.append({
                                "Timestamp": timestamp,
                                "Source": f"Simulated: {keyword}",
                                "Text": t,
                                "Sentiment": s,
                                "Confidence_Val": c,
                                "Confidence": f"{c:.1%}"
                            })
                        st.session_state.twitter_data = {"keyword": keyword, "tweets": mock_tweets}
                        st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Mini Statistics Summary in Left Panel
    if st.session_state.history:
        df_summary = pd.DataFrame(st.session_state.history)
        st.markdown('<p class="section-header">📉 Session Summary</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(df_summary))
        c2.metric("Pos%", f"{(len(df_summary[df_summary['Sentiment']=='Positive'])/len(df_summary)):.0%}")
        c3.metric("Avg Conf", f"{df_summary['Confidence_Val'].mean():.1%}")

# --- RIGHT PANEL: ANALYTICS ---
with col_right:
    st.markdown('<p class="section-header">📊 Visual Intelligence Dashboard</p>', unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.info("Ingest data on the left to activate the analytics engine.")
        # Removed broken GIF link
    else:
        df_viz = pd.DataFrame(st.session_state.history)
        
        # Row 1: Distribution & Trends
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            st.plotly_chart(create_sentiment_pie(df_viz), use_container_width=True)
        with r1_c2:
            st.plotly_chart(create_sentiment_bar(df_viz), use_container_width=True)
            
        # Row 2: Trend Line & Confidence
        r2_c1, r2_c2 = st.columns(2)
        with r2_c1:
            trend_fig = create_trend_graph(df_viz)
            if trend_fig: st.plotly_chart(trend_fig, use_container_width=True)
        with r2_c2:
            conf_fig = create_confidence_dist(df_viz)
            if conf_fig: st.plotly_chart(conf_fig, use_container_width=True)

        # Row 3: Word Clouds (High Impact)
        st.markdown("### ☁️ Linguistic Analysis (Word Clouds)")
        cloud_pos, cloud_neg = st.columns(2)
        
        with cloud_pos:
            st.markdown("<p style='text-align:center; color:#4ade80;'>Positive Vocabulary</p>", unsafe_allow_html=True)
            pos_text = " ".join(df_viz[df_viz['Sentiment'] == 'Positive']['Text'])
            fig_pos = generate_wordcloud(pos_text)
            if fig_pos: st.pyplot(fig_pos)
            else: st.write("Insufficient positive data.")
            
        with cloud_neg:
            st.markdown("<p style='text-align:center; color:#f87171;'>Negative Vocabulary</p>", unsafe_allow_html=True)
            neg_text = " ".join(df_viz[df_viz['Sentiment'] == 'Negative']['Text'])
            fig_neg = generate_wordcloud(neg_text, color='blue')
            if fig_neg: st.pyplot(fig_neg)
            else: st.write("Insufficient negative data.")

        # Twitter Specific Analytics (if data exists)
        if st.session_state.twitter_data:
            st.markdown("### 🐦 Twitter Stream Analytics")
            tw_c1, tw_c2 = st.columns(2)
            with tw_c1:
                h_fig = create_hashtag_chart(st.session_state.twitter_data['tweets'])
                if h_fig: st.plotly_chart(h_fig, use_container_width=True)
            with tw_c2:
                st.markdown("**Top Positive Tweets**")
                # Show top 2 highest confidence positive tweets
                top_pos = df_viz[df_viz['Sentiment'] == 'Positive'].sort_values('Confidence_Val', ascending=False).head(2)
                for i, row in top_pos.iterrows():
                    st.success(f"\"{row['Text'][:100]}...\"")

# --- BOTTOM PANEL: HISTORY TABLE ---
st.markdown("---")
st.markdown('<p class="section-header">📜 Detailed Activity Logs</p>', unsafe_allow_html=True)
if st.session_state.history:
    df_full = pd.DataFrame(st.session_state.history)
    st.dataframe(
        df_full[['Timestamp', 'Source', 'Sentiment', 'Confidence', 'Text']].sort_values('Timestamp', ascending=False),
        use_container_width=True,
        hide_index=True
    )
    if st.button("🗑️ Reset Dashboard Data", use_container_width=True):
        st.session_state.history = []
        st.session_state.twitter_data = None
        st.rerun()

# Footer
st.markdown("""
    <div style="text-align: center; color: #334155; font-size: 0.8rem; margin: 3rem 0;">
        PROFESSIONAL AI SENTIMENT ANALYTICS SUITE • DESIGNED FOR FINAL YEAR PORTFOLIO • &copy; 2026
    </div>
""", unsafe_allow_html=True)

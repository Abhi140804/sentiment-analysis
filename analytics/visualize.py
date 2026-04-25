import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import streamlit as st

def create_sentiment_pie(df):
    """
    Creates a professional Pie chart for sentiment distribution.
    """
    counts = df['Sentiment'].value_counts()
    color_map = {"Positive": "#4ade80", "Neutral": "#fbbf24", "Negative": "#f87171"}
    
    fig = px.pie(
        names=counts.index,
        values=counts.values,
        hole=0.4,
        color=counts.index,
        color_discrete_map=color_map,
        title="Overall Sentiment Distribution"
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "size": 14},
        showlegend=True,
        margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig

def create_sentiment_bar(df):
    """
    Creates a Bar chart for sentiment counts.
    """
    counts = df['Sentiment'].value_counts()
    color_map = {"Positive": "#4ade80", "Neutral": "#fbbf24", "Negative": "#f87171"}
    
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        color=counts.index,
        color_discrete_map=color_map,
        title="Sentiment Volume",
        labels={'x': 'Category', 'y': 'Count'}
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        showlegend=False
    )
    return fig

def create_trend_graph(df):
    """
    Creates a line graph showing sentiment trends over time.
    """
    if 'Timestamp' not in df.columns or df.empty:
        return None
        
    # Map sentiments to numeric values for trending
    # We'll use a 5-point moving average or just raw counts over time
    df = df.copy()
    # Sort by time
    df = df.sort_values('Timestamp')
    
    # We can count sentiments over time buckets if we have many entries
    # Or just plot individual points with a score
    score_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
    df['Score'] = df['Sentiment'].map(score_map)
    
    fig = px.line(
        df, x='Timestamp', y='Score',
        title="Sentiment Momentum Over Time",
        markers=True,
        color_discrete_sequence=['#38bdf8']
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        yaxis=dict(tickvals=[-1, 0, 1], ticktext=["Negative", "Neutral", "Positive"])
    )
    return fig

def create_confidence_dist(df):
    """
    Histogram showing distribution of model confidence.
    """
    if 'Confidence_Val' not in df.columns or df.empty:
        return None
        
    fig = px.histogram(
        df, x='Confidence_Val',
        nbins=20,
        title="Model Confidence Distribution",
        color_discrete_sequence=['#818cf8'],
        labels={'Confidence_Val': 'Confidence Score'}
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}
    )
    return fig

def generate_wordcloud(text, color='white'):
    """
    Generates a WordCloud image from text.
    """
    if not text.strip():
        return None
        
    wc = WordCloud(
        background_color=None,
        mode="RGBA",
        width=800,
        height=400,
        max_words=100,
        colormap='viridis' if color == 'white' else 'cool'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    plt.tight_layout(pad=0)
    
    return fig

def create_hashtag_chart(texts):
    """
    Bar chart for most frequent hashtags.
    """
    all_text = " ".join(texts)
    hashtags = re.findall(r'#(\w+)', all_text)
    
    if not hashtags:
        return None
        
    counts = Counter(hashtags).most_common(10)
    df = pd.DataFrame(counts, columns=['Hashtag', 'Count'])
    
    fig = px.bar(
        df, x='Count', y='Hashtag', orientation='h',
        title="Trending Hashtags",
        color_discrete_sequence=['#c084fc']
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        yaxis={'categoryorder':'total ascending'}
    )
    return fig

import tweepy
import re

class TwitterFetcher:
    def __init__(self, bearer_token):
        """
        Initialize the Twitter API client.
        """
        self.client = tweepy.Client(bearer_token=bearer_token)

    def clean_tweet(self, text):
        """
        Simple cleaning logic for tweets.
        """
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\S+', '', text)
        # Remove hashtags symbol (keep text)
        text = re.sub(r'#', '', text)
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def fetch_tweets(self, keyword, max_results=10):
        """
        Fetch recent tweets based on keyword.
        """
        try:
            # Query for recent tweets (exclude retweets)
            query = f"{keyword} -is:retweet lang:en"
            tweets = self.client.search_recent_tweets(query=query, max_results=max_results)
            
            if not tweets.data:
                return []
            
            cleaned_tweets = [self.clean_tweet(tweet.text) for tweet in tweets.data]
            return [t for t in cleaned_tweets if t] # Filter out empty strings
            
        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg:
                raise Exception("Twitter API Error 403: Forbidden. Your X API Free tier might not have 'search' access. Check your App permissions in the X Developer Portal.")
            elif "401" in error_msg:
                raise Exception("Twitter API Error 401: Unauthorized. Please check if your Bearer Token is correct.")
            elif "429" in error_msg:
                raise Exception("Twitter API Error 429: Rate Limit Exceeded. Please wait before trying again.")
            else:
                raise Exception(f"Twitter API Error: {error_msg}")

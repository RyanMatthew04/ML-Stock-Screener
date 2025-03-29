import os
import datetime
import requests
import pandas as pd
import time
from io import StringIO
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from gnews import GNews
import numpy as np

# ==================== 1️⃣ Fetch Nifty 50 Stock Symbols ==================== #
URL = 'https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv'
CSV_FILE = 'artifacts/nifty_50_symbols.csv'
HEADERS = {
    'Accept': 'application/json',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/110.0 Safari/537.36'
}

def is_csv_up_to_date(file_path):
    """Checks if the CSV file was modified today"""
    if os.path.exists(file_path):
        file_date = datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).date()
        return file_date == datetime.datetime.now().date()
    return False

def fetch_nifty_50_symbols():
    """Fetches and cleans the latest Nifty 50 stock symbols."""
    if is_csv_up_to_date(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        response = requests.get(URL, headers=HEADERS)
        response.raise_for_status()
        data = StringIO(response.text)
        df = pd.read_csv(data)
        df['Company Name'] = df['Company Name'].str.replace(r'\s*Ltd\.?$', '', case=False, regex=True)
        df = df[['Company Name', 'Symbol']]
        df.to_csv(CSV_FILE, index=False)
    return df

# ==================== 2️⃣ Fetch News from Google News ==================== #
class GoogleNewsFetcher:
    def __init__(self):
        self.gn = GNews(language='en', country='IN', max_results=5)

    def search_news(self, stock_name):
        """Fetches the latest news headlines for a given stock using GNews."""
        try:
            articles = self.gn.get_news(stock_name)
            if articles:
                latest_news = articles[0]  # Get the first article
                return latest_news['title']  # Return only the headline
        except Exception as e:
            print(f"❌ GNews Error: {e}")
        return None  # Return None if no news found
# ==================== 3️⃣ Classify Sentiment using FinBERT ==================== #
MODEL_PATH = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)
model.eval()

def classify_headline(headline):
    """Classifies a headline into bearish/bullish probabilities using FinBERT."""
    if not headline:  # If no headline is found, return neutral probabilities
        return np.nan, np.nan

    inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()

    probs = torch.nn.functional.softmax(torch.tensor(logits[0]), dim=0).numpy()
    bearish_prob, bullish_prob = probs[2], probs[1]

    # Normalize between bearish and bullish (ignoring neutral)
    total = bearish_prob + bullish_prob
    if total > 0:
        bearish_prob /= total
        bullish_prob /= total

    return bearish_prob, bullish_prob

# ==================== 🔥 Run Inference and Return Final DataFrame ==================== #
def finbert():
    print("🔍 Fetching latest Nifty 50 symbols...")
    stocks_df = pd.read_csv('artifacts/nifty_50_symbols.csv')

    print("🌎 Connecting to PyGoogleNews...")
    google_news_fetcher = GoogleNewsFetcher()

    print("📰 Fetching latest headlines from Google News...")
    results = []
    for _, row in stocks_df.iterrows():
        
        stock_name = row["Company Name"]
        stock_symbol = row["Symbol"]
        print(stock_name)
        headline = google_news_fetcher.search_news(stock_name)
        bearish_prob, bullish_prob = classify_headline(headline)

        results.append({
            "Stock": stock_symbol,
            'BERT_Bearish_Probability': bearish_prob,
            'BERT_Bullish_Probability': bullish_prob,
        }), 

    final_df = pd.DataFrame(results)
    return final_df



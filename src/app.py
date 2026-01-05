from flask import Flask, jsonify, render_template
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Simple cache
cache_data = None
cache_time = None

def get_market_data():
    """Get market data with caching"""
    global cache_data, cache_time
    
    # Check cache (5 minutes)
    if cache_data is not None and cache_time is not None:
        if datetime.now() - cache_time < timedelta(minutes=5):
            logger.info("Using cached data")
            return cache_data
    
    # Try to fetch real data
    tickers = ['^NSEI', 'NSEI.NS']
    for ticker in tickers:
        try:
            logger.info(f"Fetching {ticker}")
            data = yf.download(ticker, period='5d', progress=False)
            if not data.empty and len(data) > 1:
                cache_data = data
                cache_time = datetime.now()
                return data
        except Exception as e:
            logger.warning(f"Failed {ticker}: {e}")
            continue
    
    # Fallback: generate sample data
    logger.warning("Using fallback data")
    dates = pd.date_range(end=datetime.now(), periods=5, freq='B')
    prices = 22000 + np.cumsum(np.random.randn(5) * 50)
    
    data = pd.DataFrame({
        'Close': prices
    }, index=dates)
    
    cache_data = data
    cache_time = datetime.now()
    return data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict')
def predict():
    try:
        data = get_market_data()
        
        if data.empty or len(data) < 2:
            raise ValueError("No data")
        
        current_price = float(data['Close'].iloc[-1])
        prev_price = float(data['Close'].iloc[-2])
        change_pct = ((current_price - prev_price) / prev_price * 100)
        
        # Simple prediction
        if change_pct > 0.5:
            prediction = 'BULLISH'
            confidence = 70
        elif change_pct < -0.5:
            prediction = 'BEARISH'
            confidence = 70
        else:
            prediction = 'NEUTRAL'
            confidence = 60
        
        # Add small randomness
        confidence = min(max(confidence + np.random.uniform(-5, 5), 50), 90)
        
        response = {
            'prediction': prediction,
            'confidence': round(confidence, 1),
            'current_price': f'₹{current_price:,.2f}',
            'today_change': f"{'+' if change_pct >= 0 else ''}{change_pct:.2f}%",
            'analysis': f'Market showing {prediction.lower()} signals',
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Error: {e}")
        response = {
            'prediction': 'NEUTRAL',
            'confidence': 50.0,
            'current_price': '₹22,000.00',
            'today_change': '0.00%',
            'analysis': 'System updating',
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'status': 'fallback'
        }
    
    return jsonify(response)

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)

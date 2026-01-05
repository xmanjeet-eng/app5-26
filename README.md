# Nifty 50 AI Predictor

A production-ready AI-powered Nifty 50 market prediction tool deployed on Render.

## Features
- Real-time market data from Yahoo Finance
- AI-powered technical analysis
- 5-minute data caching
- Rate limiting and error handling
- Mobile-responsive UI
- Auto-refresh capability
- Health monitoring endpoints

## Deployment on Render

### Prerequisites
- Render account
- Git repository

### Steps
1. Fork/clone this repository
2. Create a new Web Service on Render
3. Connect your repository
4. Render will automatically detect the `render.yaml` file
5. Deploy!

### Environment Variables
- `DEBUG`: Set to `false` in production
- `PORT`: Automatically set by Render
- `CACHE_TIMEOUT`: Data cache timeout in seconds (default: 300)

## API Endpoints
- `GET /` - Main interface
- `GET /api/predict` - Get prediction (rate limited)
- `GET /api/health` - Health check
- `GET /api/clear-cache` - Clear cache (debug)

## Local Development
```bash
# Clone repository
git clone <repo-url>
cd nifty-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run locally
python src/app.py

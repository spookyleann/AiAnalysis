#!/usr/bin/env python3
"""
Advanced Stock Analysis Tool - Web Version
A comprehensive stock screening and analysis application with web interface
"""

from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import threading
import json

app = Flask(__name__)

# Global variable to store scan results
scan_results = {
    'status': 'idle',
    'progress': '',
    'results': [],
    'scanning': False
}

class StockAnalyzer:
    """Core analysis engine for stock evaluation"""
    
    def __init__(self):
        pass
        
    def get_sp500_tickers(self) -> List[str]:
        """Get S&P 500 tickers from Wikipedia"""
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            df = tables[0]
            return df['Symbol'].tolist()[:100]  # Limit to 100 for speed
        except:
            # Fallback to a curated list of major stocks
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
                'LLY', 'V', 'UNH', 'XOM', 'WMT', 'JPM', 'MA', 'PG', 'JNJ', 'HD',
                'COST', 'ABBV', 'AVGO', 'MRK', 'CVX', 'KO', 'PEP', 'ADBE', 'CRM',
                'NFLX', 'AMD', 'TMO', 'CSCO', 'ACN', 'MCD', 'ABT', 'INTC', 'ORCL',
                'DIS', 'NKE', 'VZ', 'CMCSA', 'WFC', 'BMY', 'PM', 'UPS', 'T', 'MS',
                'QCOM', 'NEE', 'HON', 'LOW', 'IBM', 'RTX', 'SBUX', 'SPGI', 'BA'
            ]
    
    def calculate_momentum_score(self, ticker: str, data: pd.DataFrame) -> float:
        """Calculate momentum indicators"""
        try:
            if len(data) < 50:
                return 0
            
            # Price momentum
            current_price = data['Close'].iloc[-1]
            price_20d_ago = data['Close'].iloc[-20]
            price_50d_ago = data['Close'].iloc[-50] if len(data) >= 50 else price_20d_ago
            
            momentum_20d = (current_price / price_20d_ago - 1) * 100
            momentum_50d = (current_price / price_50d_ago - 1) * 100
            
            # Volume trend
            avg_volume_recent = data['Volume'].iloc[-10:].mean()
            avg_volume_older = data['Volume'].iloc[-50:-10].mean()
            volume_increase = (avg_volume_recent / avg_volume_older - 1) * 100 if avg_volume_older > 0 else 0
            
            # Combine scores
            momentum_score = (momentum_20d * 0.4 + momentum_50d * 0.3 + min(volume_increase, 50) * 0.3)
            
            return momentum_score
        except:
            return 0
    
    def calculate_fundamental_score(self, info: dict) -> Tuple[float, Dict]:
        """Analyze fundamental metrics"""
        score = 0
        details = {}
        
        try:
            # Revenue growth
            revenue_growth = info.get('revenueGrowth', 0)
            if revenue_growth and revenue_growth > 0.15:  # 15%+ growth
                score += 25
                details['revenue_growth'] = f"{revenue_growth*100:.1f}%"
            elif revenue_growth:
                details['revenue_growth'] = f"{revenue_growth*100:.1f}%"
                score += revenue_growth * 100
            
            # Profit margins
            profit_margin = info.get('profitMargins', 0)
            if profit_margin and profit_margin > 0.15:  # 15%+ margin
                score += 20
                details['profit_margin'] = f"{profit_margin*100:.1f}%"
            elif profit_margin:
                details['profit_margin'] = f"{profit_margin*100:.1f}%"
                score += profit_margin * 50
            
            # Return on Equity
            roe = info.get('returnOnEquity', 0)
            if roe and roe > 0.15:
                score += 15
                details['roe'] = f"{roe*100:.1f}%"
            elif roe and roe > 0:
                details['roe'] = f"{roe*100:.1f}%"
                score += roe * 50
            
            # Debt to Equity
            debt_to_equity = info.get('debtToEquity', 1000)
            if debt_to_equity and debt_to_equity < 50:
                score += 15
                details['debt_to_equity'] = f"{debt_to_equity:.1f}"
            elif debt_to_equity:
                details['debt_to_equity'] = f"{debt_to_equity:.1f}"
                score += max(0, 15 - debt_to_equity / 10)
            
            # Operating Cash Flow
            operating_cashflow = info.get('operatingCashflow', 0)
            free_cashflow = info.get('freeCashflow', 0)
            if operating_cashflow and operating_cashflow > 0 and free_cashflow and free_cashflow > 0:
                score += 15
                details['cash_flow'] = 'Positive'
            
            # Earnings growth
            earnings_growth = info.get('earningsGrowth', 0)
            if earnings_growth and earnings_growth > 0.20:  # 20%+ growth
                score += 10
                details['earnings_growth'] = f"{earnings_growth*100:.1f}%"
            elif earnings_growth:
                details['earnings_growth'] = f"{earnings_growth*100:.1f}%"
            
        except Exception as e:
            pass
        
        return min(score, 100), details
    
    def calculate_valuation_score(self, info: dict) -> Tuple[float, Dict]:
        """Analyze valuation metrics"""
        score = 50  # Start neutral
        details = {}
        
        try:
            # PEG Ratio (best indicator of value for growth)
            peg = info.get('pegRatio', None)
            if peg and 0 < peg < 1.5:
                score += 30
                details['peg_ratio'] = f"{peg:.2f}"
            elif peg and peg < 2.5:
                score += 15
                details['peg_ratio'] = f"{peg:.2f}"
            elif peg:
                details['peg_ratio'] = f"{peg:.2f}"
                score -= 10
            
            # Forward PE
            forward_pe = info.get('forwardPE', None)
            if forward_pe and forward_pe < 20:
                score += 15
                details['forward_pe'] = f"{forward_pe:.1f}"
            elif forward_pe:
                details['forward_pe'] = f"{forward_pe:.1f}"
            
            # Price to Book
            pb = info.get('priceToBook', None)
            if pb and pb < 3:
                score += 10
                details['price_to_book'] = f"{pb:.2f}"
            elif pb:
                details['price_to_book'] = f"{pb:.2f}"
            
            # Enterprise Value metrics
            ev_to_revenue = info.get('enterpriseToRevenue', None)
            if ev_to_revenue and ev_to_revenue < 5:
                score += 5
                details['ev_to_revenue'] = f"{ev_to_revenue:.1f}"
            elif ev_to_revenue:
                details['ev_to_revenue'] = f"{ev_to_revenue:.1f}"
            
        except Exception as e:
            pass
        
        return min(max(score, 0), 100), details
    
    def analyze_stock(self, ticker: str) -> Dict:
        """Comprehensive stock analysis"""
        try:
            print(f"Fetching data for {ticker}...")
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check if we got valid data
            if not info or len(info) < 5:
                print(f"  → No info data for {ticker}")
                return None
            
            # Get historical data
            hist = stock.history(period='6mo')
            
            if hist.empty or len(hist) < 20:
                print(f"  → Insufficient historical data for {ticker}")
                return None
            
            print(f"  → Got {len(hist)} days of data")
            
            # Calculate scores
            momentum_score = self.calculate_momentum_score(ticker, hist)
            fundamental_score, fundamental_details = self.calculate_fundamental_score(info)
            valuation_score, valuation_details = self.calculate_valuation_score(info)
            
            # Composite score
            composite_score = (
                momentum_score * 0.35 +
                fundamental_score * 0.40 +
                valuation_score * 0.25
            )
            
            # Get key info with safe defaults
            current_price = float(hist['Close'].iloc[-1])
            market_cap = info.get('marketCap', 0)
            
            # Handle None values safely
            analyst_target = info.get('targetMeanPrice')
            if analyst_target and analyst_target > 0:
                analyst_target = float(analyst_target)
            else:
                analyst_target = None
            
            fifty_two_week_high = info.get('fiftyTwoWeekHigh')
            if fifty_two_week_high:
                fifty_two_week_high = float(fifty_two_week_high)
            
            fifty_two_week_low = info.get('fiftyTwoWeekLow')
            if fifty_two_week_low:
                fifty_two_week_low = float(fifty_two_week_low)
            
            result = {
                'ticker': ticker,
                'name': info.get('longName', info.get('shortName', ticker)),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'current_price': current_price,
                'market_cap': float(market_cap) if market_cap else 0,
                'composite_score': float(composite_score),
                'momentum_score': float(momentum_score),
                'fundamental_score': float(fundamental_score),
                'valuation_score': float(valuation_score),
                'fundamental_details': fundamental_details,
                'valuation_details': valuation_details,
                'analyst_target': analyst_target,
                'recommendation': info.get('recommendationKey', 'N/A'),
                'fifty_two_week_high': fifty_two_week_high,
                'fifty_two_week_low': fifty_two_week_low,
            }
            
            print(f"  → Score: {composite_score:.1f} (M:{momentum_score:.0f} F:{fundamental_score:.0f} V:{valuation_score:.0f})")
            
            return result
            
        except Exception as e:
            print(f"  → ERROR analyzing {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def scan_stocks(self, tickers: List[str], min_score: float = 50):
        """Scan multiple stocks and return top candidates"""
        global scan_results
        
        results = []
        total = len(tickers)
        analyzed = 0
        errors = 0
        
        scan_results['scanning'] = True
        scan_results['status'] = 'running'
        
        for idx, ticker in enumerate(tickers):
            if not scan_results['scanning']:
                scan_results['status'] = 'stopped'
                break
                
            try:
                scan_results['progress'] = f"Analyzing {ticker}... ({idx+1}/{total}) | Found: {len(results)} | Errors: {errors}"
                result = self.analyze_stock(ticker)
                
                if result:
                    analyzed += 1
                    if result['composite_score'] >= min_score:
                        results.append(result)
                        print(f"✓ {ticker}: Score {result['composite_score']:.1f}")
                    else:
                        print(f"- {ticker}: Score {result['composite_score']:.1f} (below threshold)")
                else:
                    errors += 1
                    print(f"✗ {ticker}: Failed to analyze")
                    
            except Exception as e:
                errors += 1
                print(f"✗ {ticker}: Error - {str(e)}")
                continue
        
        # Sort by composite score
        results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        scan_results['results'] = results
        scan_results['status'] = 'complete'
        scan_results['scanning'] = False
        scan_results['progress'] = f"Complete! Analyzed: {analyzed}/{total} | Found {len(results)} stocks above {min_score} | Errors: {errors}"
        
        print(f"\n{'='*60}")
        print(f"SCAN COMPLETE")
        print(f"Total tickers: {total}")
        print(f"Successfully analyzed: {analyzed}")
        print(f"Above threshold ({min_score}): {len(results)}")
        print(f"Errors: {errors}")
        print(f"{'='*60}\n")
        
        return results


analyzer = StockAnalyzer()

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/start_scan', methods=['POST'])
def start_scan():
    """Start the stock scanning process"""
    global scan_results
    
    if scan_results['scanning']:
        return jsonify({'error': 'Scan already in progress'})
    
    min_score = request.json.get('min_score', 60)
    test_mode = request.json.get('test_mode', False)
    
    # Reset results
    scan_results = {
        'status': 'starting',
        'progress': 'Getting stock list...',
        'results': [],
        'scanning': True
    }
    
    # Start scan in background thread
    tickers = analyzer.get_sp500_tickers()
    
    # If test mode, only scan first 10
    if test_mode:
        tickers = tickers[:10]
        scan_results['progress'] = f'TEST MODE: Scanning {len(tickers)} stocks...'
    
    thread = threading.Thread(
        target=analyzer.scan_stocks,
        args=(tickers, min_score),
        daemon=True
    )
    thread.start()
    
    return jsonify({'status': 'started', 'tickers': len(tickers)})

@app.route('/stop_scan', methods=['POST'])
def stop_scan():
    """Stop the scanning process"""
    global scan_results
    scan_results['scanning'] = False
    return jsonify({'status': 'stopping'})

@app.route('/get_status')
def get_status():
    """Get current scan status"""
    return jsonify(scan_results)

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Stock Analyzer - AI-Powered Research Tool</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .controls {
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }
        
        .control-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #495057;
        }
        
        input[type="range"] {
            width: 100%;
            height: 8px;
            border-radius: 5px;
            background: #d3d3d3;
            outline: none;
            -webkit-appearance: none;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
        }
        
        .score-display {
            text-align: center;
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
            margin-top: 10px;
        }
        
        .button-group {
            display: flex;
            gap: 15px;
            margin-top: 20px;
        }
        
        button {
            flex: 1;
            padding: 15px 30px;
            font-size: 1.1em;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-primary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-danger {
            background: #dc3545;
            color: white;
        }
        
        .btn-danger:hover:not(:disabled) {
            background: #c82333;
        }
        
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
        
        .btn-secondary:hover {
            background: #5a6268;
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .progress-container {
            padding: 20px 30px;
            background: #fff3cd;
            border-bottom: 1px solid #ffc107;
            display: none;
        }
        
        .progress-text {
            font-weight: 600;
            color: #856404;
            margin-bottom: 10px;
        }
        
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #f8f9fa;
            border-radius: 15px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            transition: width 0.3s;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .content {
            padding: 30px;
        }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #dee2e6;
        }
        
        .tab {
            padding: 12px 24px;
            background: none;
            border: none;
            border-bottom: 3px solid transparent;
            cursor: pointer;
            font-weight: 600;
            color: #6c757d;
            transition: all 0.3s;
        }
        
        .tab.active {
            color: #667eea;
            border-bottom-color: #667eea;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .results {
            max-height: 600px;
            overflow-y: auto;
        }
        
        .stock-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            border-left: 5px solid #667eea;
            transition: all 0.3s;
        }
        
        .stock-card:hover {
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transform: translateX(5px);
        }
        
        .stock-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .stock-title {
            font-size: 1.3em;
            font-weight: bold;
            color: #212529;
        }
        
        .stock-ticker {
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }
        
        .stock-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .info-item {
            background: white;
            padding: 10px;
            border-radius: 5px;
        }
        
        .info-label {
            font-size: 0.9em;
            color: #6c757d;
            margin-bottom: 5px;
        }
        
        .info-value {
            font-size: 1.1em;
            font-weight: 600;
            color: #212529;
        }
        
        .score-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1.2em;
        }
        
        .score-exceptional {
            background: #28a745;
            color: white;
        }
        
        .score-strong {
            background: #17a2b8;
            color: white;
        }
        
        .score-good {
            background: #ffc107;
            color: #212529;
        }
        
        .score-average {
            background: #6c757d;
            color: white;
        }
        
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #6c757d;
        }
        
        .empty-state h2 {
            font-size: 2em;
            margin-bottom: 10px;
        }
        
        .about-section {
            line-height: 1.8;
        }
        
        .about-section h2 {
            color: #667eea;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        
        .about-section ul {
            margin-left: 20px;
        }
        
        .about-section li {
            margin-bottom: 8px;
        }
        
        .disclaimer {
            background: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .disclaimer strong {
            color: #856404;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Advanced Stock Analyzer</h1>
            <p>AI-Powered Stock Research & Screening Tool</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label for="minScore">Minimum Score Threshold:</label>
                <input type="range" id="minScore" min="0" max="100" value="40" step="5">
                <div class="score-display">40</div>
            </div>
            
            <div class="button-group">
                <button id="startBtn" class="btn-primary" onclick="startScan()">
                    🔍 Start Full Scan (100+ stocks)
                </button>
                <button id="testBtn" class="btn-primary" onclick="startScan(true)" style="background: #17a2b8;">
                    🧪 Test Mode (10 stocks)
                </button>
                <button id="stopBtn" class="btn-danger" onclick="stopScan()" disabled>
                    ⏹ Stop Scan
                </button>
                <button class="btn-secondary" onclick="clearResults()">
                    🗑 Clear Results
                </button>
            </div>
        </div>
        
        <div id="progressContainer" class="progress-container">
            <div class="progress-text" id="progressText">Ready to scan...</div>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
        </div>
        
        <div class="content">
            <div class="tabs">
                <button class="tab active" onclick="switchTab('opportunities')">📊 Top Opportunities</button>
                <button class="tab" onclick="switchTab('detailed')">🔬 Detailed Analysis</button>
                <button class="tab" onclick="switchTab('about')">ℹ️ About</button>
            </div>
            
            <div id="opportunities" class="tab-content active">
                <div id="resultsContainer" class="results">
                    <div class="empty-state">
                        <h2>Ready to Find Opportunities</h2>
                        <p>Click "Start Stock Scan" to analyze 100+ major stocks</p>
                    </div>
                </div>
            </div>
            
            <div id="detailed" class="tab-content">
                <div id="detailedContainer" class="results">
                    <div class="empty-state">
                        <h2>No Analysis Yet</h2>
                        <p>Run a scan to see detailed metrics</p>
                    </div>
                </div>
            </div>
            
            <div id="about" class="tab-content">
                <div class="about-section">
                    <div class="disclaimer">
                        <strong>⚠️ IMPORTANT DISCLAIMER</strong><br>
                        This tool is for RESEARCH purposes only. It does NOT provide financial advice.
                        Past performance does not guarantee future results. Always do your own research
                        and consider consulting a financial advisor before making investment decisions.
                    </div>
                    
                    <h2>How It Works</h2>
                    <p>This tool analyzes stocks using a multi-factor scoring system:</p>
                    <ul>
                        <li><strong>Momentum (35%):</strong> Price trends and volume analysis</li>
                        <li><strong>Fundamentals (40%):</strong> Revenue growth, profit margins, ROE, debt levels, cash flow</li>
                        <li><strong>Valuation (25%):</strong> PEG ratio, P/E ratios, price-to-book, enterprise value metrics</li>
                    </ul>
                    
                    <h2>Score Interpretation</h2>
                    <ul>
                        <li><strong>80-100:</strong> Exceptional opportunity (very rare)</li>
                        <li><strong>70-79:</strong> Strong candidate</li>
                        <li><strong>60-69:</strong> Above average</li>
                        <li><strong>50-59:</strong> Average/neutral</li>
                        <li><strong>Below 50:</strong> Below average</li>
                    </ul>
                    
                    <h2>What This Tool Does</h2>
                    <ul>
                        <li>Scans 100+ major stocks automatically</li>
                        <li>Analyzes multiple financial metrics</li>
                        <li>Provides composite scoring</li>
                        <li>Identifies stocks worth further research</li>
                        <li>Compares current price to analyst targets</li>
                    </ul>
                    
                    <h2>What This Tool Does NOT Do</h2>
                    <ul>
                        <li>Predict the future with certainty</li>
                        <li>Guarantee profits</li>
                        <li>Replace your own research</li>
                        <li>Work for day trading strategies</li>
                        <li>Constitute professional financial advice</li>
                    </ul>
                    
                    <h2>Next Steps After Finding Stocks</h2>
                    <ol>
                        <li>Read the company's SEC filings (10-K, 10-Q)</li>
                        <li>Watch recent earnings calls</li>
                        <li>Understand the business model</li>
                        <li>Check recent news</li>
                        <li>Compare to competitors</li>
                        <li>Assess competitive advantages</li>
                        <li>Consider if it fits your investment strategy</li>
                    </ol>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let pollingInterval = null;
        
        // Score slider
        document.getElementById('minScore').addEventListener('input', function(e) {
            document.querySelector('.score-display').textContent = e.target.value;
        });
        
        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }
        
        function startScan(testMode = false) {
            const minScore = document.getElementById('minScore').value;
            
            document.getElementById('startBtn').disabled = true;
            document.getElementById('testBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('progressContainer').style.display = 'block';
            
            fetch('/start_scan', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    min_score: parseInt(minScore),
                    test_mode: testMode
                })
            });
            
            // Start polling for status
            pollingInterval = setInterval(checkStatus, 1000);
        }
        
        function stopScan() {
            fetch('/stop_scan', {method: 'POST'});
            clearInterval(pollingInterval);
            document.getElementById('startBtn').disabled = false;
            document.getElementById('testBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        }
        
        function clearResults() {
            document.getElementById('resultsContainer').innerHTML = `
                <div class="empty-state">
                    <h2>Ready to Find Opportunities</h2>
                    <p>Click "Start Stock Scan" to analyze 100+ major stocks</p>
                </div>
            `;
            document.getElementById('detailedContainer').innerHTML = `
                <div class="empty-state">
                    <h2>No Analysis Yet</h2>
                    <p>Run a scan to see detailed metrics</p>
                </div>
            `;
        }
        
        function checkStatus() {
            fetch('/get_status')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('progressText').textContent = data.progress;
                    
                    if (data.status === 'complete') {
                        clearInterval(pollingInterval);
                        document.getElementById('startBtn').disabled = false;
                        document.getElementById('testBtn').disabled = false;
                        document.getElementById('stopBtn').disabled = true;
                        displayResults(data.results);
                    }
                    
                    if (data.status === 'stopped') {
                        clearInterval(pollingInterval);
                        document.getElementById('startBtn').disabled = false;
                        document.getElementById('testBtn').disabled = false;
                        document.getElementById('stopBtn').disabled = true;
                    }
                });
        }
        
        function displayResults(results) {
            if (results.length === 0) {
                document.getElementById('resultsContainer').innerHTML = `
                    <div class="empty-state">
                        <h2>No Stocks Found</h2>
                        <p>Try lowering the minimum score threshold</p>
                    </div>
                `;
                return;
            }
            
            // Summary view
            let summaryHTML = '';
            results.slice(0, 20).forEach((stock, idx) => {
                const scoreClass = 
                    stock.composite_score >= 80 ? 'score-exceptional' :
                    stock.composite_score >= 70 ? 'score-strong' :
                    stock.composite_score >= 60 ? 'score-good' : 'score-average';
                
                const upside = stock.analyst_target ? 
                    ((stock.analyst_target / stock.current_price - 1) * 100).toFixed(1) : null;
                
                summaryHTML += `
                    <div class="stock-card">
                        <div class="stock-header">
                            <div class="stock-title">${idx + 1}. ${stock.name}</div>
                            <div class="stock-ticker">${stock.ticker}</div>
                        </div>
                        <div>
                            <span class="score-badge ${scoreClass}">
                                ${stock.composite_score.toFixed(1)} / 100
                            </span>
                        </div>
                        <div class="stock-info">
                            <div class="info-item">
                                <div class="info-label">Sector</div>
                                <div class="info-value">${stock.sector}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Current Price</div>
                                <div class="info-value">$${stock.current_price.toFixed(2)}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Market Cap</div>
                                <div class="info-value">$${(stock.market_cap / 1e9).toFixed(1)}B</div>
                            </div>
                            ${upside ? `
                            <div class="info-item">
                                <div class="info-label">Analyst Upside</div>
                                <div class="info-value">${upside > 0 ? '+' : ''}${upside}%</div>
                            </div>
                            ` : ''}
                            <div class="info-item">
                                <div class="info-label">Momentum</div>
                                <div class="info-value">${stock.momentum_score.toFixed(1)}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Fundamentals</div>
                                <div class="info-value">${stock.fundamental_score.toFixed(1)}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Valuation</div>
                                <div class="info-value">${stock.valuation_score.toFixed(1)}</div>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            document.getElementById('resultsContainer').innerHTML = summaryHTML;
            
            // Detailed view
            let detailedHTML = '';
            results.slice(0, 20).forEach((stock, idx) => {
                detailedHTML += `
                    <div class="stock-card">
                        <h3>#${idx + 1} - ${stock.ticker}: ${stock.name}</h3>
                        <p><strong>Sector:</strong> ${stock.sector} | <strong>Industry:</strong> ${stock.industry}</p>
                        <p><strong>Current Price:</strong> $${stock.current_price.toFixed(2)} | 
                           <strong>Market Cap:</strong> $${(stock.market_cap / 1e9).toFixed(2)}B</p>
                        
                        <h4 style="margin-top: 15px;">Scores</h4>
                        <p>🎯 <strong>Composite:</strong> ${stock.composite_score.toFixed(1)}/100</p>
                        <p>📈 <strong>Momentum:</strong> ${stock.momentum_score.toFixed(1)}/100</p>
                        <p>💼 <strong>Fundamentals:</strong> ${stock.fundamental_score.toFixed(1)}/100</p>
                        <p>💰 <strong>Valuation:</strong> ${stock.valuation_score.toFixed(1)}/100</p>
                        
                        <h4 style="margin-top: 15px;">Fundamental Metrics</h4>
                        ${Object.entries(stock.fundamental_details).map(([key, value]) => 
                            `<p>• <strong>${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:</strong> ${value}</p>`
                        ).join('')}
                        
                        <h4 style="margin-top: 15px;">Valuation Metrics</h4>
                        ${Object.entries(stock.valuation_details).map(([key, value]) => 
                            `<p>• <strong>${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:</strong> ${value}</p>`
                        ).join('')}
                        
                        ${stock.analyst_target ? `
                        <h4 style="margin-top: 15px;">Analyst Consensus</h4>
                        <p><strong>Target Price:</strong> $${stock.analyst_target.toFixed(2)}</p>
                        <p><strong>Potential Upside:</strong> ${((stock.analyst_target / stock.current_price - 1) * 100).toFixed(1)}%</p>
                        <p><strong>Recommendation:</strong> ${stock.recommendation.toUpperCase()}</p>
                        ` : ''}
                    </div>
                `;
            });
            
            document.getElementById('detailedContainer').innerHTML = detailedHTML;
        }
    </script>
</body>
</html>
'''

# Create templates directory and save HTML
import os
os.makedirs('templates', exist_ok=True)
with open('templates/index.html', 'w') as f:
    f.write(HTML_TEMPLATE)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Advanced Stock Analyzer - Web Version")
    print("="*60)
    print("\nStarting web server...")
    print("\n📱 Open your browser and go to:")
    print("\n    http://localhost:5000")
    print("\n⏹  Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=False, host='127.0.0.1', port=5000)
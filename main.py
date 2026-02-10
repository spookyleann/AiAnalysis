#!/usr/bin/env python3
“””
Advanced Stock Analysis Tool
A comprehensive stock screening and analysis application
“””

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
import json
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import requests

class StockAnalyzer:
“”“Core analysis engine for stock evaluation”””

```
def __init__(self):
    self.results_queue = queue.Queue()
    
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
            'NFLX', 'AMD', 'TMO', 'CSCO', 'ACN', 'MCD', 'ABT', 'INTC', 'ORCL'
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
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get historical data
        hist = stock.history(period='6mo')
        
        if hist.empty or len(hist) < 20:
            return None
        
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
        
        # Get key info
        current_price = hist['Close'].iloc[-1]
        market_cap = info.get('marketCap', 0)
        
        result = {
            'ticker': ticker,
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'current_price': current_price,
            'market_cap': market_cap,
            'composite_score': composite_score,
            'momentum_score': momentum_score,
            'fundamental_score': fundamental_score,
            'valuation_score': valuation_score,
            'fundamental_details': fundamental_details,
            'valuation_details': valuation_details,
            'analyst_target': info.get('targetMeanPrice', None),
            'recommendation': info.get('recommendationKey', 'N/A'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', None),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', None),
        }
        
        return result
        
    except Exception as e:
        return None

def scan_stocks(self, tickers: List[str], progress_callback, min_score: float = 50):
    """Scan multiple stocks and return top candidates"""
    results = []
    total = len(tickers)
    
    for idx, ticker in enumerate(tickers):
        try:
            progress_callback(f"Analyzing {ticker}... ({idx+1}/{total})")
            result = self.analyze_stock(ticker)
            
            if result and result['composite_score'] >= min_score:
                results.append(result)
                
        except Exception as e:
            continue
    
    # Sort by composite score
    results.sort(key=lambda x: x['composite_score'], reverse=True)
    return results
```

class StockAnalyzerGUI:
“”“Main GUI Application”””

```
def __init__(self, root):
    self.root = root
    self.root.title("Advanced Stock Analyzer - AI-Powered Research Tool")
    self.root.geometry("1200x800")
    
    self.analyzer = StockAnalyzer()
    self.scanning = False
    
    self.setup_ui()
    
def setup_ui(self):
    """Create the user interface"""
    
    # Title
    title_frame = ttk.Frame(self.root, padding="10")
    title_frame.pack(fill=tk.X)
    
    title_label = ttk.Label(
        title_frame,
        text="🚀 Advanced Stock Analyzer",
        font=('Helvetica', 20, 'bold')
    )
    title_label.pack()
    
    subtitle = ttk.Label(
        title_frame,
        text="AI-Powered Stock Research & Screening Tool",
        font=('Helvetica', 10)
    )
    subtitle.pack()
    
    # Control Panel
    control_frame = ttk.LabelFrame(self.root, text="Scan Controls", padding="10")
    control_frame.pack(fill=tk.X, padx=10, pady=5)
    
    # Minimum score slider
    score_frame = ttk.Frame(control_frame)
    score_frame.pack(fill=tk.X, pady=5)
    
    ttk.Label(score_frame, text="Minimum Score:").pack(side=tk.LEFT, padx=5)
    self.min_score_var = tk.IntVar(value=60)
    self.min_score_slider = ttk.Scale(
        score_frame,
        from_=0,
        to=100,
        variable=self.min_score_var,
        orient=tk.HORIZONTAL,
        length=300
    )
    self.min_score_slider.pack(side=tk.LEFT, padx=5)
    self.score_label = ttk.Label(score_frame, text="60")
    self.score_label.pack(side=tk.LEFT, padx=5)
    
    def update_score_label(event=None):
        self.score_label.config(text=str(self.min_score_var.get()))
    
    self.min_score_slider.configure(command=update_score_label)
    
    # Buttons
    button_frame = ttk.Frame(control_frame)
    button_frame.pack(fill=tk.X, pady=5)
    
    self.scan_button = ttk.Button(
        button_frame,
        text="🔍 Start Stock Scan",
        command=self.start_scan,
        style='Accent.TButton'
    )
    self.scan_button.pack(side=tk.LEFT, padx=5)
    
    self.stop_button = ttk.Button(
        button_frame,
        text="⏹ Stop Scan",
        command=self.stop_scan,
        state=tk.DISABLED
    )
    self.stop_button.pack(side=tk.LEFT, padx=5)
    
    self.clear_button = ttk.Button(
        button_frame,
        text="🗑 Clear Results",
        command=self.clear_results
    )
    self.clear_button.pack(side=tk.LEFT, padx=5)
    
    # Progress
    self.progress_var = tk.StringVar(value="Ready to scan...")
    self.progress_label = ttk.Label(control_frame, textvariable=self.progress_var)
    self.progress_label.pack(pady=5)
    
    self.progress_bar = ttk.Progressbar(control_frame, mode='indeterminate')
    self.progress_bar.pack(fill=tk.X, pady=5)
    
    # Results area with notebook
    self.notebook = ttk.Notebook(self.root)
    self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    # Tab 1: Top Opportunities
    self.results_frame = ttk.Frame(self.notebook)
    self.notebook.add(self.results_frame, text="📊 Top Opportunities")
    
    # Results text area
    self.results_text = scrolledtext.ScrolledText(
        self.results_frame,
        wrap=tk.WORD,
        font=('Courier', 10),
        height=30
    )
    self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Tab 2: Detailed Analysis
    self.detail_frame = ttk.Frame(self.notebook)
    self.notebook.add(self.detail_frame, text="🔬 Detailed Analysis")
    
    self.detail_text = scrolledtext.ScrolledText(
        self.detail_frame,
        wrap=tk.WORD,
        font=('Courier', 9),
        height=30
    )
    self.detail_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Tab 3: About
    about_frame = ttk.Frame(self.notebook)
    self.notebook.add(about_frame, text="ℹ️ About")
    
    about_text = scrolledtext.ScrolledText(about_frame, wrap=tk.WORD, font=('Helvetica', 10))
    about_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    about_text.insert('1.0', self.get_about_text())
    about_text.config(state=tk.DISABLED)
    
    # Status bar
    self.status_var = tk.StringVar(value="Ready")
    status_bar = ttk.Label(
        self.root,
        textvariable=self.status_var,
        relief=tk.SUNKEN,
        anchor=tk.W
    )
    status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
def get_about_text(self):
    """Return about information"""
    return """
```

ADVANCED STOCK ANALYZER v1.0
AI-Powered Research & Screening Tool

FEATURES:
• Multi-factor analysis combining momentum, fundamentals, and valuation
• Automated screening of major stocks
• Composite scoring system (0-100 scale)
• Real-time data from Yahoo Finance

SCORING METHODOLOGY:

Composite Score = (35% Momentum + 40% Fundamentals + 25% Valuation)

1. MOMENTUM SCORE (35%):
- 20-day and 50-day price momentum
- Volume trend analysis
- Relative strength indicators
1. FUNDAMENTAL SCORE (40%):
- Revenue growth (15%+ ideal)
- Profit margins (15%+ ideal)
- Return on equity
- Debt-to-equity ratio
- Cash flow strength
- Earnings growth
1. VALUATION SCORE (25%):
- PEG ratio (growth-adjusted P/E)
- Forward P/E ratio
- Price-to-book ratio
- Enterprise value metrics

INTERPRETATION:
• 80-100: Exceptional opportunity (rare)
• 70-79: Strong candidate
• 60-69: Above average
• 50-59: Average/Neutral
• Below 50: Below average

IMPORTANT DISCLAIMERS:
• This tool is for RESEARCH purposes only
• Past performance does not guarantee future results
• No algorithm can predict stock movements with certainty
• Always do your own due diligence
• Consider consulting a financial advisor
• Markets are inherently unpredictable

DATA SOURCE:
Yahoo Finance (yfinance library)
Data may be delayed and should be verified from official sources.

This tool helps you identify stocks worth researching further - it does NOT
tell you what to buy. Use it as a starting point for deeper analysis.
“””

```
def start_scan(self):
    """Start the stock scanning process"""
    if self.scanning:
        return
    
    self.scanning = True
    self.scan_button.config(state=tk.DISABLED)
    self.stop_button.config(state=tk.NORMAL)
    self.progress_bar.start()
    self.clear_results()
    
    # Start scanning in separate thread
    thread = threading.Thread(target=self.run_scan, daemon=True)
    thread.start()
    
def stop_scan(self):
    """Stop the scanning process"""
    self.scanning = False
    self.scan_button.config(state=tk.NORMAL)
    self.stop_button.config(state=tk.DISABLED)
    self.progress_bar.stop()
    self.progress_var.set("Scan stopped by user")
    
def run_scan(self):
    """Execute the scan in background"""
    try:
        min_score = self.min_score_var.get()
        tickers = self.analyzer.get_sp500_tickers()
        
        def progress_callback(message):
            if not self.scanning:
                raise Exception("Scan stopped")
            self.root.after(0, self.update_progress, message)
        
        results = self.analyzer.scan_stocks(tickers, progress_callback, min_score)
        
        self.root.after(0, self.display_results, results)
        
    except Exception as e:
        self.root.after(0, self.show_error, str(e))
    finally:
        self.root.after(0, self.scan_complete)

def update_progress(self, message):
    """Update progress display"""
    self.progress_var.set(message)
    
def display_results(self, results):
    """Display scan results"""
    self.results_text.delete('1.0', tk.END)
    self.detail_text.delete('1.0', tk.END)
    
    if not results:
        self.results_text.insert('1.0', "No stocks met the minimum score criteria.\n\n")
        self.results_text.insert(tk.END, "Try lowering the minimum score threshold.")
        return
    
    # Summary view
    header = f"{'='*80}\n"
    header += f"TOP STOCK OPPORTUNITIES - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    header += f"Found {len(results)} stocks meeting criteria (min score: {self.min_score_var.get()})\n"
    header += f"{'='*80}\n\n"
    
    self.results_text.insert('1.0', header)
    
    for idx, stock in enumerate(results[:20], 1):  # Top 20
        summary = f"{idx}. {stock['ticker']} - {stock['name']}\n"
        summary += f"   Sector: {stock['sector']} | Industry: {stock['industry']}\n"
        summary += f"   Price: ${stock['current_price']:.2f} | "
        summary += f"Market Cap: ${stock['market_cap']/1e9:.1f}B\n"
        summary += f"   📊 COMPOSITE SCORE: {stock['composite_score']:.1f}/100\n"
        summary += f"   • Momentum: {stock['momentum_score']:.1f} | "
        summary += f"Fundamentals: {stock['fundamental_score']:.1f} | "
        summary += f"Valuation: {stock['valuation_score']:.1f}\n"
        
        if stock['analyst_target']:
            upside = (stock['analyst_target'] / stock['current_price'] - 1) * 100
            summary += f"   • Analyst Target: ${stock['analyst_target']:.2f} ({upside:+.1f}% upside)\n"
        
        summary += "\n" + "-"*80 + "\n\n"
        
        self.results_text.insert(tk.END, summary)
    
    # Detailed view
    detail_header = f"{'='*100}\n"
    detail_header += f"DETAILED ANALYSIS - TOP {min(len(results), 20)} STOCKS\n"
    detail_header += f"{'='*100}\n\n"
    
    self.detail_text.insert('1.0', detail_header)
    
    for idx, stock in enumerate(results[:20], 1):
        detail = f"\n{'='*100}\n"
        detail += f"#{idx} - {stock['ticker']}: {stock['name']}\n"
        detail += f"{'='*100}\n\n"
        
        detail += f"OVERVIEW:\n"
        detail += f"  Sector: {stock['sector']}\n"
        detail += f"  Industry: {stock['industry']}\n"
        detail += f"  Current Price: ${stock['current_price']:.2f}\n"
        detail += f"  Market Cap: ${stock['market_cap']/1e9:.2f}B\n"
        detail += f"  52-Week Range: ${stock['fifty_two_week_low']:.2f} - ${stock['fifty_two_week_high']:.2f}\n\n"
        
        detail += f"SCORES:\n"
        detail += f"  🎯 Composite Score: {stock['composite_score']:.1f}/100\n"
        detail += f"  📈 Momentum Score: {stock['momentum_score']:.1f}/100\n"
        detail += f"  💼 Fundamental Score: {stock['fundamental_score']:.1f}/100\n"
        detail += f"  💰 Valuation Score: {stock['valuation_score']:.1f}/100\n\n"
        
        detail += f"FUNDAMENTAL METRICS:\n"
        for key, value in stock['fundamental_details'].items():
            detail += f"  • {key.replace('_', ' ').title()}: {value}\n"
        
        detail += f"\nVALUATION METRICS:\n"
        for key, value in stock['valuation_details'].items():
            detail += f"  • {key.replace('_', ' ').title()}: {value}\n"
        
        if stock['analyst_target']:
            detail += f"\nANALYST CONSENSUS:\n"
            detail += f"  Target Price: ${stock['analyst_target']:.2f}\n"
            upside = (stock['analyst_target'] / stock['current_price'] - 1) * 100
            detail += f"  Potential Upside: {upside:+.1f}%\n"
            detail += f"  Recommendation: {stock['recommendation'].upper()}\n"
        
        detail += "\n"
        
        self.detail_text.insert(tk.END, detail)
    
    self.status_var.set(f"Scan complete - Found {len(results)} opportunities")
    
def scan_complete(self):
    """Clean up after scan completes"""
    self.scanning = False
    self.scan_button.config(state=tk.NORMAL)
    self.stop_button.config(state=tk.DISABLED)
    self.progress_bar.stop()
    self.progress_var.set("Scan complete!")
    
def clear_results(self):
    """Clear all results"""
    self.results_text.delete('1.0', tk.END)
    self.detail_text.delete('1.0', tk.END)
    self.status_var.set("Results cleared")
    
def show_error(self, message):
    """Show error message"""
    messagebox.showerror("Error", f"An error occurred:\n{message}")
    self.progress_var.set("Error occurred")
```

def main():
“”“Main entry point”””
root = tk.Tk()

```
# Style configuration
style = ttk.Style()
style.theme_use('default')

app = StockAnalyzerGUI(root)
root.mainloop()
```

if **name** == “**main**”:
main()
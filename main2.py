#!/usr/bin/env python3
"""
Real-Time Trading Analyzer for Gold Futures (GC) and NASDAQ Futures (NQ/MNQ)
Uses Alpha Vantage API + News Sentiment Analysis + Futures-Specific Analysis
Provides buy/sell signals, risk assessment, and TP/SL recommendations
"""

import sys
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import argparse
import time

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  Installing required packages...")
    print("Run: pip3 install pandas numpy requests")
    print("\nContinuing with demo mode...\n")


class AlphaVantageAPI:
    """Alpha Vantage API handler - FREE API for stock/commodity data"""
    
    def __init__(self, api_key: str = "demo"):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        
    def get_intraday_data(self, symbol: str, interval: str = "5min") -> Optional[pd.DataFrame]:
        """Fetch intraday data"""
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'apikey': self.api_key,
            'outputsize': 'full'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if 'Error Message' in data:
                print(f"❌ API Error: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                print(f"⚠️  API Limit: {data['Note']}")
                return None
                
            time_series_key = f'Time Series ({interval})'
            if time_series_key not in data:
                print(f"❌ No data available. Using demo data...")
                return None
            
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })
            df = df.astype(float)
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def get_daily_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch daily data"""
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'full'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if 'Time Series (Daily)' not in data:
                return None
            
            df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })
            df = df.astype(float)
            df = df.sort_index()
            
            return df.tail(200)  # Last 200 days
            
        except Exception as e:
            print(f"❌ Error fetching daily data: {e}")
            return None
    
    def get_news_sentiment(self, tickers: List[str], topics: List[str] = None) -> List[Dict]:
        """Fetch news and sentiment data"""
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ','.join(tickers),
            'apikey': self.api_key,
            'limit': 50
        }
        
        if topics:
            params['topics'] = ','.join(topics)
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if 'feed' in data:
                return data['feed']
            return []
            
        except Exception as e:
            print(f"❌ Error fetching news: {e}")
            return []


class NewsAPI:
    """News API for additional news sources"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
    
    def get_news(self, query: str, days_back: int = 7) -> List[Dict]:
        """Fetch news articles"""
        if not self.api_key:
            return []
        
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        params = {
            'q': query,
            'from': from_date,
            'sortBy': 'relevancy',
            'apiKey': self.api_key,
            'language': 'en',
            'pageSize': 50
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if data.get('status') == 'ok':
                return data.get('articles', [])
            return []
            
        except Exception as e:
            print(f"⚠️  News API error: {e}")
            return []


class SentimentAnalyzer:
    """Analyze news sentiment"""
    
    BULLISH_KEYWORDS = [
        'rally', 'surge', 'boom', 'growth', 'gain', 'rise', 'bull', 'bullish',
        'breakthrough', 'record high', 'soar', 'jump', 'climb', 'advance',
        'positive', 'optimistic', 'strength', 'strong', 'upgrade', 'beat expectations',
        'profit', 'revenue growth', 'expansion', 'innovation', 'success', 'outperform',
        'momentum', 'breakout', 'bullish trend', 'buying opportunity', 'accumulation'
    ]
    
    BEARISH_KEYWORDS = [
        'crash', 'fall', 'decline', 'drop', 'plunge', 'bear', 'bearish',
        'recession', 'downturn', 'slump', 'tumble', 'lose', 'loss', 'losses',
        'negative', 'pessimistic', 'weakness', 'weak', 'downgrade', 'miss expectations',
        'risk', 'concern', 'worry', 'fear', 'crisis', 'trouble', 'problem', 'underperform',
        'correction', 'selloff', 'sell-off', 'resistance', 'distribution', 'overbought'
    ]
    
    @classmethod
    def analyze_text(cls, text: str) -> Tuple[float, str]:
        """Analyze sentiment of text. Returns (score, label)"""
        if not text:
            return 0.0, "Neutral"
        
        text_lower = text.lower()
        
        bullish_count = sum(1 for word in cls.BULLISH_KEYWORDS if word in text_lower)
        bearish_count = sum(1 for word in cls.BEARISH_KEYWORDS if word in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0, "Neutral"
        
        # Score from -1 (very bearish) to +1 (very bullish)
        score = (bullish_count - bearish_count) / total
        
        if score > 0.3:
            label = "Bullish"
        elif score < -0.3:
            label = "Bearish"
        else:
            label = "Neutral"
        
        return score, label
    
    @classmethod
    def analyze_news_feed(cls, news_articles: List[Dict]) -> Dict:
        """Analyze multiple news articles"""
        if not news_articles:
            return {
                'overall_sentiment': 0.0,
                'sentiment_label': 'Neutral',
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'total_articles': 0,
                'recent_headlines': []
            }
        
        sentiments = []
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        recent_headlines = []
        
        for article in news_articles[:20]:  # Analyze top 20
            title = article.get('title', '') or article.get('headline', '')
            summary = article.get('description', '') or article.get('summary', '')
            text = f"{title} {summary}"
            
            score, label = cls.analyze_text(text)
            sentiments.append(score)
            
            if label == "Bullish":
                bullish_count += 1
            elif label == "Bearish":
                bearish_count += 1
            else:
                neutral_count += 1
            
            if len(recent_headlines) < 5:
                recent_headlines.append({
                    'title': title,
                    'sentiment': label,
                    'score': round(score, 2),
                    'time': article.get('time_published', article.get('publishedAt', ''))
                })
        
        overall_sentiment = np.mean(sentiments) if sentiments else 0.0
        
        if overall_sentiment > 0.2:
            sentiment_label = "Bullish"
        elif overall_sentiment < -0.2:
            sentiment_label = "Bearish"
        else:
            sentiment_label = "Neutral"
        
        return {
            'overall_sentiment': round(overall_sentiment, 3),
            'sentiment_label': sentiment_label,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'total_articles': len(news_articles),
            'recent_headlines': recent_headlines,
            'sentiment_score_distribution': {
                'mean': round(np.mean(sentiments), 3) if sentiments else 0,
                'std': round(np.std(sentiments), 3) if sentiments else 0
            }
        }


class TechnicalAnalyzer:
    """Technical analysis calculations"""
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Avoid division by zero
        loss = loss.replace(0, 0.0001)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    @staticmethod
    def calculate_macd(data: pd.Series) -> Tuple[float, float, float]:
        """Calculate MACD, Signal, and Histogram"""
        exp1 = data.ewm(span=12, adjust=False).mean()
        exp2 = data.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd.iloc[-1], signal.iloc[-1], histogram.iloc[-1]
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper.iloc[-1], sma.iloc[-1], lower.iloc[-1]
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> float:
        """Calculate Exponential Moving Average"""
        ema = data.ewm(span=period, adjust=False).mean()
        return ema.iloc[-1]
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> float:
        """Calculate Simple Moving Average"""
        sma = data.rolling(window=period).mean()
        return sma.iloc[-1]
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range"""
        high_low = high - low
        high_close = abs(high - close.shift())
        low_close = abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        denominator = highest_high - lowest_low
        denominator = denominator.replace(0, 0.0001)
        
        k = 100 * ((close - lowest_low) / denominator)
        d = k.rolling(window=3).mean()
        return k.iloc[-1], d.iloc[-1]
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Average Directional Index"""
        high_diff = high.diff()
        low_diff = -low.diff()
        
        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        atr = TechnicalAnalyzer.calculate_atr(high, low, close, period)
        atr = max(atr, 0.0001)  # Avoid division by zero
        
        pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)
        
        di_sum = pos_di + neg_di
        di_sum = di_sum.replace(0, 0.0001)
        
        dx = 100 * abs(pos_di - neg_di) / di_sum
        adx = dx.rolling(window=period).mean()
        
        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25.0
    
    @staticmethod
    def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> float:
        """Calculate Volume Weighted Average Price (important for futures)"""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap.iloc[-1] if not pd.isna(vwap.iloc[-1]) else close.iloc[-1]


class FuturesAnalyzer:
    """Futures-specific analysis for MNQ/NQ and GC traders"""
    
    @staticmethod
    def calculate_tick_value(asset_type: str) -> Dict:
        """Calculate tick size and value for futures contracts"""
        futures_specs = {
            'mnq': {
                'name': 'Micro E-mini NASDAQ-100',
                'tick_size': 0.25,
                'tick_value': 0.50,  # $0.50 per tick
                'contract_multiplier': 2,
                'margin_approx': 1500,  # Approximate initial margin
                'exchange': 'CME'
            },
            'nq': {
                'name': 'E-mini NASDAQ-100',
                'tick_size': 0.25,
                'tick_value': 5.00,  # $5.00 per tick
                'contract_multiplier': 20,
                'margin_approx': 15000,
                'exchange': 'CME'
            },
            'gc': {
                'name': 'Gold Futures',
                'tick_size': 0.10,
                'tick_value': 10.00,  # $10.00 per tick
                'contract_multiplier': 100,  # 100 troy ounces
                'margin_approx': 10000,
                'exchange': 'COMEX'
            },
            'mgc': {
                'name': 'Micro Gold Futures',
                'tick_size': 0.10,
                'tick_value': 1.00,  # $1.00 per tick
                'contract_multiplier': 10,  # 10 troy ounces
                'margin_approx': 1000,
                'exchange': 'COMEX'
            }
        }
        return futures_specs.get(asset_type.lower(), futures_specs['mnq'])
    
    @staticmethod
    def calculate_position_sizing(price: float, atr: float, account_size: float, risk_percent: float, futures_specs: Dict) -> Dict:
        """Calculate position sizing for futures based on risk management"""
        risk_amount = account_size * (risk_percent / 100)
        
        # Calculate points at risk (using ATR as stop distance)
        points_at_risk = atr
        
        # Calculate dollar risk per contract
        dollar_risk_per_contract = points_at_risk * futures_specs['contract_multiplier']
        
        # Calculate number of contracts
        max_contracts = int(risk_amount / dollar_risk_per_contract)
        max_contracts = max(1, max_contracts)  # At least 1 contract
        
        # Calculate actual risk with this position
        actual_risk = max_contracts * dollar_risk_per_contract
        
        # Calculate ticks at risk
        ticks_at_risk = points_at_risk / futures_specs['tick_size']
        
        return {
            'max_contracts': max_contracts,
            'risk_per_contract': round(dollar_risk_per_contract, 2),
            'total_risk': round(actual_risk, 2),
            'risk_percentage': round((actual_risk / account_size) * 100, 2),
            'points_at_risk': round(points_at_risk, 2),
            'ticks_at_risk': round(ticks_at_risk, 0),
            'margin_required': max_contracts * futures_specs['margin_approx']
        }


class TradingSignalGenerator:
    """Generate trading signals based on technical analysis and news sentiment"""
    
    def __init__(self, symbol: str, data: pd.DataFrame, news_sentiment: Dict, asset_type: str = 'mnq'):
        self.symbol = symbol
        self.data = data
        self.news_sentiment = news_sentiment
        self.asset_type = asset_type
        self.current_price = data['Close'].iloc[-1]
        self.signals = []
        self.score = 0
        self.futures_specs = FuturesAnalyzer.calculate_tick_value(asset_type)
        
    def analyze(self) -> Dict:
        """Perform comprehensive analysis"""
        close = self.data['Close']
        high = self.data['High']
        low = self.data['Low']
        volume = self.data['Volume']
        
        # Calculate all indicators
        rsi = TechnicalAnalyzer.calculate_rsi(close)
        macd, macd_signal, macd_hist = TechnicalAnalyzer.calculate_macd(close)
        bb_upper, bb_middle, bb_lower = TechnicalAnalyzer.calculate_bollinger_bands(close)
        ema_9 = TechnicalAnalyzer.calculate_ema(close, 9)
        ema_21 = TechnicalAnalyzer.calculate_ema(close, 21)
        ema_50 = TechnicalAnalyzer.calculate_ema(close, 50)
        sma_200 = TechnicalAnalyzer.calculate_sma(close, 200)
        atr = TechnicalAnalyzer.calculate_atr(high, low, close)
        stoch_k, stoch_d = TechnicalAnalyzer.calculate_stochastic(high, low, close)
        adx = TechnicalAnalyzer.calculate_adx(high, low, close)
        vwap = TechnicalAnalyzer.calculate_vwap(high, low, close, volume)
        
        # Analyze each indicator
        self._analyze_rsi(rsi)
        self._analyze_macd(macd, macd_signal, macd_hist)
        self._analyze_bollinger(bb_upper, bb_middle, bb_lower)
        self._analyze_moving_averages(ema_9, ema_21, ema_50, sma_200)
        self._analyze_stochastic(stoch_k, stoch_d)
        self._analyze_trend_strength(adx)
        self._analyze_volume(volume)
        self._analyze_vwap(vwap)  # Important for futures day trading
        
        # Analyze news sentiment
        self._analyze_news_sentiment()
        
        # Calculate TP and SL (futures-adjusted)
        tp_sl = self._calculate_tp_sl_futures(atr, ema_21)
        
        # Calculate position sizing
        position_sizing = self._calculate_position_sizing(atr)
        
        # Determine overall signal
        signal = self._determine_signal()
        risk_level = self._assess_risk(adx, atr)
        
        return {
            'symbol': self.symbol,
            'asset_type': self.asset_type.upper(),
            'futures_name': self.futures_specs['name'],
            'current_price': round(self.current_price, 2),
            'signal': signal,
            'score': self.score,
            'max_score': 16,  # Updated max score with VWAP
            'risk_level': risk_level,
            'indicators': {
                'RSI': round(rsi, 2),
                'MACD': round(macd, 4),
                'MACD_Signal': round(macd_signal, 4),
                'MACD_Histogram': round(macd_hist, 4),
                'BB_Upper': round(bb_upper, 2),
                'BB_Middle': round(bb_middle, 2),
                'BB_Lower': round(bb_lower, 2),
                'EMA_9': round(ema_9, 2),
                'EMA_21': round(ema_21, 2),
                'EMA_50': round(ema_50, 2),
                'SMA_200': round(sma_200, 2),
                'ATR': round(atr, 2),
                'Stochastic_K': round(stoch_k, 2),
                'Stochastic_D': round(stoch_d, 2),
                'ADX': round(adx, 2),
                'VWAP': round(vwap, 2)
            },
            'futures_specs': self.futures_specs,
            'news_sentiment': self.news_sentiment,
            'tp_sl': tp_sl,
            'position_sizing': position_sizing,
            'signals': self.signals
        }
    
    def _analyze_vwap(self, vwap: float):
        """Analyze VWAP - critical for futures day trading"""
        price = self.current_price
        
        if price > vwap:
            diff_pct = ((price - vwap) / vwap) * 100
            self.signals.append(f"📊 VWAP: Price above VWAP (${vwap:.2f}, +{diff_pct:.2f}%) - Bullish bias")
            self.score += 1
        else:
            diff_pct = ((vwap - price) / vwap) * 100
            self.signals.append(f"📊 VWAP: Price below VWAP (${vwap:.2f}, -{diff_pct:.2f}%) - Bearish bias")
            self.score -= 1
    
    def _calculate_position_sizing(self, atr: float) -> List[Dict]:
        """Calculate position sizing for different account sizes"""
        account_sizes = [5000, 10000, 25000, 50000, 100000]
        risk_percent = 1.0  # 1% risk per trade
        
        sizing = []
        for account_size in account_sizes:
            size_calc = FuturesAnalyzer.calculate_position_sizing(
                self.current_price, atr, account_size, risk_percent, self.futures_specs
            )
            sizing.append({
                'account_size': account_size,
                **size_calc
            })
        
        return sizing
    
    def _analyze_news_sentiment(self):
        """Analyze news sentiment and add to score"""
        sentiment = self.news_sentiment.get('overall_sentiment', 0)
        sentiment_label = self.news_sentiment.get('sentiment_label', 'Neutral')
        total_articles = self.news_sentiment.get('total_articles', 0)
        
        if total_articles == 0:
            self.signals.append("📰 News: No recent news articles found")
            return
        
        # Sentiment can add up to ±3 points
        if sentiment > 0.4:
            self.signals.append(f"📰 News: VERY BULLISH sentiment (+{sentiment:.2f}) - Strong buy signal")
            self.score += 3
        elif sentiment > 0.2:
            self.signals.append(f"📰 News: Bullish sentiment (+{sentiment:.2f}) - Buy signal")
            self.score += 2
        elif sentiment > 0:
            self.signals.append(f"📰 News: Slightly bullish (+{sentiment:.2f})")
            self.score += 1
        elif sentiment < -0.4:
            self.signals.append(f"📰 News: VERY BEARISH sentiment ({sentiment:.2f}) - Strong sell signal")
            self.score -= 3
        elif sentiment < -0.2:
            self.signals.append(f"📰 News: Bearish sentiment ({sentiment:.2f}) - Sell signal")
            self.score -= 2
        elif sentiment < 0:
            self.signals.append(f"📰 News: Slightly bearish ({sentiment:.2f})")
            self.score -= 1
        else:
            self.signals.append(f"📰 News: Neutral sentiment - No clear direction")
        
        # Add article count info
        bullish = self.news_sentiment.get('bullish_count', 0)
        bearish = self.news_sentiment.get('bearish_count', 0)
        neutral = self.news_sentiment.get('neutral_count', 0)
        self.signals.append(f"   Articles analyzed: {total_articles} (📈{bullish} bullish, 📉{bearish} bearish, ➖{neutral} neutral)")
    
    def _analyze_rsi(self, rsi: float):
        """Analyze RSI indicator"""
        if rsi < 30:
            self.signals.append("📊 RSI: OVERSOLD - Strong buy signal")
            self.score += 2
        elif rsi < 40:
            self.signals.append("📊 RSI: Approaching oversold - Bullish")
            self.score += 1
        elif rsi > 70:
            self.signals.append("📊 RSI: OVERBOUGHT - Strong sell signal")
            self.score -= 2
        elif rsi > 60:
            self.signals.append("📊 RSI: Approaching overbought - Bearish")
            self.score -= 1
        else:
            self.signals.append("📊 RSI: Neutral zone")
    
    def _analyze_macd(self, macd: float, signal: float, hist: float):
        """Analyze MACD indicator"""
        if macd > signal and hist > 0:
            self.signals.append("📈 MACD: Bullish crossover - Buy signal")
            self.score += 2
        elif macd < signal and hist < 0:
            self.signals.append("📉 MACD: Bearish crossover - Sell signal")
            self.score -= 2
        elif macd > signal:
            self.signals.append("📈 MACD: Above signal line - Bullish")
            self.score += 1
        else:
            self.signals.append("📉 MACD: Below signal line - Bearish")
            self.score -= 1
    
    def _analyze_bollinger(self, upper: float, middle: float, lower: float):
        """Analyze Bollinger Bands"""
        price = self.current_price
        if price <= lower:
            self.signals.append("💎 Price at lower Bollinger Band - Oversold")
            self.score += 1
        elif price >= upper:
            self.signals.append("⚠️  Price at upper Bollinger Band - Overbought")
            self.score -= 1
        else:
            band_position = (price - lower) / (upper - lower) * 100
            if band_position < 30:
                self.signals.append(f"📊 Bollinger: Lower {band_position:.0f}% - Bullish area")
            elif band_position > 70:
                self.signals.append(f"📊 Bollinger: Upper {band_position:.0f}% - Bearish area")
    
    def _analyze_moving_averages(self, ema9: float, ema21: float, ema50: float, sma200: float):
        """Analyze moving averages"""
        price = self.current_price
        
        # Golden/Death Cross
        if ema50 > sma200:
            self.signals.append("🌟 Golden Cross (EMA50 > SMA200) - Long-term bullish")
            self.score += 1
        elif ema50 < sma200:
            self.signals.append("☠️  Death Cross (EMA50 < SMA200) - Long-term bearish")
            self.score -= 1
        
        # Short-term trend
        if price > ema9 > ema21 > ema50:
            self.signals.append("🚀 Strong uptrend - All EMAs aligned bullish")
            self.score += 2
        elif price < ema9 < ema21 < ema50:
            self.signals.append("⬇️  Strong downtrend - All EMAs aligned bearish")
            self.score -= 2
        elif price > ema21:
            self.signals.append("📈 Price above EMA21 - Short-term bullish")
            self.score += 1
        else:
            self.signals.append("📉 Price below EMA21 - Short-term bearish")
            self.score -= 1
    
    def _analyze_stochastic(self, k: float, d: float):
        """Analyze Stochastic Oscillator"""
        if k < 20 and d < 20:
            self.signals.append("⚡ Stochastic: Oversold zone - Buy opportunity")
            self.score += 1
        elif k > 80 and d > 80:
            self.signals.append("⚡ Stochastic: Overbought zone - Sell opportunity")
            self.score -= 1
    
    def _analyze_trend_strength(self, adx: float):
        """Analyze trend strength using ADX"""
        if adx > 50:
            self.signals.append(f"💪 ADX: {adx:.1f} - Very strong trend")
        elif adx > 25:
            self.signals.append(f"📊 ADX: {adx:.1f} - Strong trend")
        else:
            self.signals.append(f"〰️  ADX: {adx:.1f} - Weak/no trend (range-bound)")
    
    def _analyze_volume(self, volume: pd.Series):
        """Analyze volume trends"""
        recent_vol = volume.iloc[-1]
        avg_vol = volume.iloc[-20:].mean()
        
        if recent_vol > avg_vol * 1.5:
            self.signals.append("📢 High volume - Strong momentum")
            self.score += 1
        elif recent_vol < avg_vol * 0.5:
            self.signals.append("🔇 Low volume - Weak momentum")
    
    def _calculate_tp_sl_futures(self, atr: float, ema21: float) -> Dict:
        """Calculate Take Profit and Stop Loss levels for futures"""
        price = self.current_price
        tick_size = self.futures_specs['tick_size']
        tick_value = self.futures_specs['tick_value']
        
        # Round to nearest tick
        def round_to_tick(value):
            return round(value / tick_size) * tick_size
        
        # Conservative, Moderate, and Aggressive strategies
        strategies = {
            'scalping': {
                'name': 'Scalping (Quick In/Out)',
                'tp_long': round_to_tick(price + (atr * 0.5)),
                'sl_long': round_to_tick(price - (atr * 0.3)),
                'tp_short': round_to_tick(price - (atr * 0.5)),
                'sl_short': round_to_tick(price + (atr * 0.3)),
                'risk_reward': 1.67
            },
            'conservative': {
                'name': 'Conservative Day Trade',
                'tp_long': round_to_tick(price + (atr * 1.5)),
                'sl_long': round_to_tick(price - (atr * 1.0)),
                'tp_short': round_to_tick(price - (atr * 1.5)),
                'sl_short': round_to_tick(price + (atr * 1.0)),
                'risk_reward': 1.5
            },
            'moderate': {
                'name': 'Moderate Swing',
                'tp_long': round_to_tick(price + (atr * 2.0)),
                'sl_long': round_to_tick(price - (atr * 1.0)),
                'tp_short': round_to_tick(price - (atr * 2.0)),
                'sl_short': round_to_tick(price + (atr * 1.0)),
                'risk_reward': 2.0
            },
            'aggressive': {
                'name': 'Aggressive Position',
                'tp_long': round_to_tick(price + (atr * 3.0)),
                'sl_long': round_to_tick(price - (atr * 1.0)),
                'tp_short': round_to_tick(price - (atr * 3.0)),
                'sl_short': round_to_tick(price + (atr * 1.0)),
                'risk_reward': 3.0
            }
        }
        
        # Add calculations for each strategy
        for strategy in strategies.values():
            # Long position calculations
            strategy['tp_long_pct'] = ((strategy['tp_long'] - price) / price) * 100
            strategy['sl_long_pct'] = ((price - strategy['sl_long']) / price) * 100
            strategy['tp_long_ticks'] = (strategy['tp_long'] - price) / tick_size
            strategy['sl_long_ticks'] = (price - strategy['sl_long']) / tick_size
            strategy['tp_long_dollars'] = strategy['tp_long_ticks'] * tick_value
            strategy['sl_long_dollars'] = strategy['sl_long_ticks'] * tick_value
            
            # Short position calculations
            strategy['tp_short_pct'] = ((price - strategy['tp_short']) / price) * 100
            strategy['sl_short_pct'] = ((strategy['sl_short'] - price) / price) * 100
            strategy['tp_short_ticks'] = (price - strategy['tp_short']) / tick_size
            strategy['sl_short_ticks'] = (strategy['sl_short'] - price) / tick_size
            strategy['tp_short_dollars'] = strategy['tp_short_ticks'] * tick_value
            strategy['sl_short_dollars'] = strategy['sl_short_ticks'] * tick_value
        
        return strategies
    
    def _determine_signal(self) -> str:
        """Determine overall trading signal"""
        if self.score >= 8:
            return "STRONG BUY 🚀"
        elif self.score >= 4:
            return "BUY 📈"
        elif self.score <= -8:
            return "STRONG SELL 📉"
        elif self.score <= -4:
            return "SELL ⬇️"
        else:
            return "HOLD ⏸️"
    
    def _assess_risk(self, adx: float, atr: float) -> str:
        """Assess risk level"""
        price = self.current_price
        volatility = (atr / price) * 100
        
        # Strong trend with high volatility = higher risk
        if adx > 40 and volatility > 3:
            return "HIGH ⚠️"
        elif adx > 25 and volatility > 2:
            return "MEDIUM ⚡"
        elif volatility < 1.5:
            return "LOW ✅"
        else:
            return "MEDIUM ⚡"


class TradingAnalyzer:
    """Main trading analyzer class"""
    
    def __init__(self, alpha_vantage_key: str = "demo", news_api_key: str = None):
        self.av_api = AlphaVantageAPI(alpha_vantage_key)
        self.news_api = NewsAPI(news_api_key)
        self.assets = {
            'mnq': {
                'symbol': 'QQQ',  # NASDAQ-100 ETF (tracks what MNQ follows)
                'name': 'Micro E-mini NASDAQ-100 Futures',
                'type': 'mnq'
            },
            'nq': {
                'symbol': 'QQQ',  # NASDAQ-100 ETF
                'name': 'E-mini NASDAQ-100 Futures',
                'type': 'nq'
            },
            'gold': {
                'symbol': 'GLD',  # SPDR Gold Trust ETF
                'name': 'Gold Futures',
                'type': 'gc'
            },
            'gc': {
                'symbol': 'GLD',  # Gold Futures
                'name': 'Gold Futures (Full Contract)',
                'type': 'gc'
            },
            'mgc': {
                'symbol': 'GLD',  # Micro Gold Futures
                'name': 'Micro Gold Futures',
                'type': 'mgc'
            }
        }
        self.news_queries = {
            'mnq': 'NASDAQ OR "NASDAQ-100" OR "tech stocks" OR "technology futures"',
            'nq': 'NASDAQ OR "NASDAQ-100" OR "tech stocks" OR "technology futures"',
            'gold': 'gold OR "gold futures" OR "gold prices" OR "precious metals"',
            'gc': 'gold OR "gold futures" OR "gold prices" OR "precious metals"',
            'mgc': 'gold OR "gold futures" OR "gold prices" OR "precious metals"'
        }
    
    def fetch_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical data"""
        if not PANDAS_AVAILABLE:
            return None
        
        print(f"   Fetching market data for {symbol}...")
        data = self.av_api.get_daily_data(symbol)
        
        if data is None or len(data) < 50:
            print(f"   ⚠️  Using demo data for {symbol}")
            return self._generate_demo_data()
        
        return data
    
    def fetch_news(self, asset_name: str) -> List[Dict]:
        """Fetch news for asset"""
        print(f"   Fetching news and sentiment...")
        
        asset_info = self.assets.get(asset_name.lower())
        if not asset_info:
            return []
        
        symbol = asset_info['symbol']
        news = self.av_api.get_news_sentiment([symbol])
        
        # Try News API as backup
        if not news and self.news_api.api_key:
            query = self.news_queries.get(asset_name.lower(), asset_name)
            news = self.news_api.get_news(query)
        
        return news
    
    def analyze_asset(self, asset_name: str) -> Optional[Dict]:
        """Analyze a specific asset"""
        asset_info = self.assets.get(asset_name.lower())
        if not asset_info:
            print(f"❌ Unknown asset: {asset_name}")
            print(f"Available: {', '.join(self.assets.keys())}")
            return None
        
        symbol = asset_info['symbol']
        asset_type = asset_info['type']
        full_name = asset_info['name']
        
        print(f"\n{'='*70}")
        print(f"🔍 Analyzing {full_name}")
        print(f"   Symbol: {symbol} | Futures Type: {asset_type.upper()}")
        print(f"{'='*70}")
        
        # Fetch data
        data = self.fetch_data(symbol)
        if data is None:
            return None
        
        # Fetch and analyze news
        news = self.fetch_news(asset_name)
        news_sentiment = SentimentAnalyzer.analyze_news_feed(news)
        
        # Generate signals
        signal_gen = TradingSignalGenerator(symbol, data, news_sentiment, asset_type)
        analysis = signal_gen.analyze()
        
        return analysis
    
    def _generate_demo_data(self) -> pd.DataFrame:
        """Generate demo data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
        np.random.seed(42)
        
        close = 100 + np.cumsum(np.random.randn(200) * 2)
        high = close + np.random.rand(200) * 2
        low = close - np.random.rand(200) * 2
        open_price = close + np.random.randn(200)
        volume = np.random.randint(1000000, 10000000, 200)
        
        df = pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        }, index=dates)
        
        return df
    
    def display_analysis(self, analysis: Dict):
        """Display analysis results in a formatted way"""
        print(f"\n📊 Current Price: ${analysis['current_price']:,.2f}")
        print(f"📋 Contract: {analysis['futures_name']}")
        
        specs = analysis['futures_specs']
        print(f"💼 Futures Specs:")
        print(f"   Tick Size: ${specs['tick_size']} | Tick Value: ${specs['tick_value']:.2f}")
        print(f"   Multiplier: {specs['contract_multiplier']}x | Approx Margin: ${specs['margin_approx']:,}")
        
        print(f"\n{'='*70}")
        print(f"🎯 TRADING SIGNAL: {analysis['signal']}")
        print(f"📊 Signal Strength: {analysis['score']}/{analysis['max_score']}")
        print(f"⚠️  Risk Level: {analysis['risk_level']}")
        print(f"{'='*70}")
        
        # News Sentiment Summary
        news = analysis['news_sentiment']
        if news.get('total_articles', 0) > 0:
            print(f"\n📰 NEWS SENTIMENT ANALYSIS:")
            print(f"{'-'*70}")
            print(f"  Overall Sentiment: {news['sentiment_label']} ({news['overall_sentiment']:+.3f})")
            print(f"  Articles Analyzed: {news['total_articles']}")
            print(f"  Breakdown: 📈 {news['bullish_count']} Bullish | " 
                  f"📉 {news['bearish_count']} Bearish | " 
                  f"➖ {news['neutral_count']} Neutral")
            
            if news.get('recent_headlines'):
                print(f"\n  Recent Headlines:")
                for i, headline in enumerate(news['recent_headlines'], 1):
                    sentiment_emoji = "📈" if headline['sentiment'] == "Bullish" else "📉" if headline['sentiment'] == "Bearish" else "➖"
                    print(f"    {i}. {sentiment_emoji} {headline['title'][:80]}...")
                    print(f"       Sentiment: {headline['sentiment']} ({headline['score']:+.2f})")
        
        print(f"\n📈 TECHNICAL INDICATORS:")
        print(f"{'-'*70}")
        indicators = analysis['indicators']
        print(f"  RSI (14):        {indicators['RSI']:>8.2f}  {'(Oversold)' if indicators['RSI'] < 30 else '(Overbought)' if indicators['RSI'] > 70 else '(Neutral)'}")
        print(f"  MACD:            {indicators['MACD']:>8.4f}")
        print(f"  MACD Signal:     {indicators['MACD_Signal']:>8.4f}")
        print(f"  MACD Histogram:  {indicators['MACD_Histogram']:>8.4f}")
        print(f"\n  Bollinger Bands:")
        print(f"    Upper:         ${indicators['BB_Upper']:>8,.2f}")
        print(f"    Middle:        ${indicators['BB_Middle']:>8,.2f}")
        print(f"    Lower:         ${indicators['BB_Lower']:>8,.2f}")
        print(f"\n  Moving Averages:")
        print(f"    EMA 9:         ${indicators['EMA_9']:>8,.2f}")
        print(f"    EMA 21:        ${indicators['EMA_21']:>8,.2f}")
        print(f"    EMA 50:        ${indicators['EMA_50']:>8,.2f}")
        print(f"    SMA 200:       ${indicators['SMA_200']:>8,.2f}")
        print(f"    VWAP:          ${indicators['VWAP']:>8,.2f}  ⭐ (Day Trading Key Level)")
        print(f"\n  Volatility & Momentum:")
        print(f"    ATR (14):      {indicators['ATR']:>8.2f}")
        print(f"    Stochastic %K: {indicators['Stochastic_K']:>8.2f}")
        print(f"    Stochastic %D: {indicators['Stochastic_D']:>8.2f}")
        print(f"    ADX (14):      {indicators['ADX']:>8.2f}")
        
        print(f"\n💡 SIGNAL BREAKDOWN:")
        print(f"{'-'*70}")
        for signal in analysis['signals']:
            print(f"  {signal}")
        
        print(f"\n💰 POSITION SIZING (1% Risk Per Trade):")
        print(f"{'-'*70}")
        for sizing in analysis['position_sizing'][:3]:  # Show first 3
            print(f"\n  Account Size: ${sizing['account_size']:,}")
            print(f"    Max Contracts:    {sizing['max_contracts']}")
            print(f"    Risk/Contract:    ${sizing['risk_per_contract']:,.2f}")
            print(f"    Total Risk:       ${sizing['total_risk']:,.2f} ({sizing['risk_percentage']:.2f}%)")
            print(f"    Margin Required:  ${sizing['margin_required']:,}")
            print(f"    Points at Risk:   {sizing['points_at_risk']:.2f} (~{int(sizing['ticks_at_risk'])} ticks)")
        
        print(f"\n🎯 TAKE PROFIT & STOP LOSS RECOMMENDATIONS:")
        print(f"{'-'*70}")
        
        tp_sl = analysis['tp_sl']
        
        for key, strategy in tp_sl.items():
            icon = "⚡" if key == "scalping" else "💚" if key == "conservative" else "💛" if key == "moderate" else "❤️"
            print(f"\n  {icon} {strategy['name'].upper()} (R:R = {strategy['risk_reward']}:1):")
            print(f"    LONG Position:")
            print(f"      Take Profit:  ${strategy['tp_long']:>8,.2f}  (+{strategy['tp_long_pct']:.2f}% | {int(strategy['tp_long_ticks'])} ticks | ${strategy['tp_long_dollars']:.2f})")
            print(f"      Stop Loss:    ${strategy['sl_long']:>8,.2f}  (-{strategy['sl_long_pct']:.2f}% | {int(strategy['sl_long_ticks'])} ticks | ${strategy['sl_long_dollars']:.2f})")
            print(f"    SHORT Position:")
            print(f"      Take Profit:  ${strategy['tp_short']:>8,.2f}  (+{strategy['tp_short_pct']:.2f}% | {int(strategy['tp_short_ticks'])} ticks | ${strategy['tp_short_dollars']:.2f})")
            print(f"      Stop Loss:    ${strategy['sl_short']:>8,.2f}  (-{strategy['sl_short_pct']:.2f}% | {int(strategy['sl_short_ticks'])} ticks | ${strategy['sl_short_dollars']:.2f})")
        
        print(f"\n{'='*70}")
        print(f"⚠️  DISCLAIMER: This is for educational purposes only.")
        print(f"   Futures trading involves substantial risk of loss.")
        print(f"   Always do your own research and consult a financial advisor.")
        print(f"{'='*70}\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Real-Time Futures Trading Analyzer for MNQ, NQ, and Gold with News Sentiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 trading_analyzer.py --asset mnq --av-key YOUR_KEY
  python3 trading_analyzer.py --asset nq --av-key YOUR_KEY
  python3 trading_analyzer.py --asset gold --av-key YOUR_KEY
  python3 trading_analyzer.py --asset gc --av-key YOUR_KEY
  python3 trading_analyzer.py --asset mnq --av-key YOUR_KEY --news-key YOUR_NEWS_KEY

Available Assets:
  mnq   - Micro E-mini NASDAQ-100 Futures (what you trade!)
  nq    - E-mini NASDAQ-100 Futures
  gold  - Gold Futures
  gc    - Gold Futures (full contract)
  mgc   - Micro Gold Futures

API Keys (Free):
  Alpha Vantage: https://www.alphavantage.co/support/#api-key
  News API (optional): https://newsapi.org/register
        """
    )
    
    parser.add_argument(
        '--asset',
        choices=['mnq', 'nq', 'gold', 'gc', 'mgc'],
        default='mnq',
        help='Futures contract to analyze (default: mnq)'
    )
    
    parser.add_argument(
        '--av-key',
        default='demo',
        help='Alpha Vantage API key (get free at alphavantage.co)'
    )
    
    parser.add_argument(
        '--news-key',
        help='News API key (optional, for additional news sources)'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results in JSON format'
    )
    
    args = parser.parse_args()
    
    if not PANDAS_AVAILABLE:
        print("\n" + "="*70)
        print("❌ REQUIRED PACKAGES NOT INSTALLED")
        print("="*70)
        print("\nTo use this tool, install the required packages:")
        print("\n  pip3 install pandas numpy requests")
        print("\nOr if you have permission issues:")
        print("\n  pip3 install --user pandas numpy requests")
        print("\n" + "="*70 + "\n")
        sys.exit(1)
    
    if args.av_key == 'demo':
        print("\n" + "="*70)
        print("📌 USING DEMO API KEY")
        print("="*70)
        print("\nYou're using the demo API key which has strict rate limits.")
        print("For better results, get a FREE API key at:")
        print("  https://www.alphavantage.co/support/#api-key")
        print("\nThen run:")
        print(f"  python3 trading_analyzer.py --asset {args.asset} --av-key YOUR_API_KEY")
        print("="*70 + "\n")
    
    analyzer = TradingAnalyzer(args.av_key, args.news_key)
    
    analysis = analyzer.analyze_asset(args.asset)
    if analysis:
        if not args.json:
            analyzer.display_analysis(analysis)
        else:
            print(json.dumps(analysis, indent=2, default=str))


if __name__ == '__main__':
    main()
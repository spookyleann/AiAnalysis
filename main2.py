#!/usr/bin/env python3
"""
Live Futures Trading Analyzer with Real-Time Prices
Interactive terminal interface for MNQ, NQ, Gold futures
Includes technical analysis strategies and AI-powered predictions
"""

import sys
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  Required packages not installed!")
    print("Run: pip3 install pandas numpy requests")
    sys.exit(1)

# API KEYS - EDIT THESE!
ALPHA_VANTAGE_KEY = "9VOO4G3GU4K2PMOF"  # ← PUT YOUR KEY HERE
NEWS_API_KEY = "cbed9b4d8cea4e6b83a7755078db25dd"  # ← OPTIONAL: PUT YOUR NEWS API KEY HERE


class LiveDataFetcher:
    """Fetch REAL-TIME live prices from multiple sources"""
    
    def __init__(self, av_key: str):
        self.av_key = av_key
        
    def get_live_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote from Alpha Vantage"""
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.av_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'Global Quote' in data and data['Global Quote']:
                quote = data['Global Quote']
                return {
                    'symbol': quote.get('01. symbol'),
                    'price': float(quote.get('05. price', 0)),
                    'change': float(quote.get('09. change', 0)),
                    'change_pct': quote.get('10. change percent', '0%').replace('%', ''),
                    'volume': int(float(quote.get('06. volume', 0))),
                    'latest_day': quote.get('07. latest trading day'),
                    'open': float(quote.get('02. open', 0)),
                    'high': float(quote.get('03. high', 0)),
                    'low': float(quote.get('04. low', 0)),
                    'prev_close': float(quote.get('08. previous close', 0))
                }
        except Exception as e:
            print(f"❌ Error fetching live quote: {e}")
        
        return None
    
    def get_historical_data(self, symbol: str, outputsize: str = 'compact') -> Optional[pd.DataFrame]:
        """Get historical daily data"""
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': outputsize,
            'apikey': self.av_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
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
                return df
        except Exception as e:
            print(f"❌ Error fetching historical data: {e}")
        
        return None


class FuturesConverter:
    """Convert ETF prices to actual futures prices with live market data"""
    
    FUTURES_MAP = {
        'MNQ': {
            'name': 'Micro E-mini NASDAQ-100',
            'proxy': 'QQQ',
            'multiplier': 41.35,  # Approximate QQQ to NQ conversion
            'tick_size': 0.25,
            'tick_value': 0.50,
            'margin': 1500,
            'exchange': 'CME',
            'hours': '18:00-17:00 EST (Sun-Fri)'
        },
        'NQ': {
            'name': 'E-mini NASDAQ-100',
            'proxy': 'QQQ',
            'multiplier': 41.35,
            'tick_size': 0.25,
            'tick_value': 5.00,
            'margin': 15000,
            'exchange': 'CME',
            'hours': '18:00-17:00 EST (Sun-Fri)'
        },
        'GC': {
            'name': 'Gold Futures',
            'proxy': 'GLD',
            'multiplier': 18.5,  # GLD is ~1/10th of gold price, futures are 100oz
            'tick_size': 0.10,
            'tick_value': 10.00,
            'margin': 10000,
            'exchange': 'COMEX',
            'hours': '18:00-17:00 EST (Sun-Fri)'
        },
        'MGC': {
            'name': 'Micro Gold Futures',
            'proxy': 'GLD',
            'multiplier': 1.85,  # 10oz vs 100oz
            'tick_size': 0.10,
            'tick_value': 1.00,
            'margin': 1000,
            'exchange': 'COMEX',
            'hours': '18:00-17:00 EST (Sun-Fri)'
        },
        'ES': {
            'name': 'E-mini S&P 500',
            'proxy': 'SPY',
            'multiplier': 12.5,
            'tick_size': 0.25,
            'tick_value': 12.50,
            'margin': 12000,
            'exchange': 'CME',
            'hours': '18:00-17:00 EST (Sun-Fri)'
        },
        'MES': {
            'name': 'Micro E-mini S&P 500',
            'proxy': 'SPY',
            'multiplier': 12.5,
            'tick_size': 0.25,
            'tick_value': 1.25,
            'margin': 1200,
            'exchange': 'CME',
            'hours': '18:00-17:00 EST (Sun-Fri)'
        }
    }
    
    @classmethod
    def get_futures_info(cls, symbol: str) -> Optional[Dict]:
        """Get futures contract information"""
        return cls.FUTURES_MAP.get(symbol.upper())
    
    @classmethod
    def convert_price(cls, etf_price: float, symbol: str) -> float:
        """Convert ETF price to futures price"""
        info = cls.get_futures_info(symbol)
        if info:
            return etf_price * info['multiplier']
        return etf_price
    
    @classmethod
    def list_available_symbols(cls) -> List[str]:
        """List all available futures symbols"""
        return list(cls.FUTURES_MAP.keys())


class TechnicalIndicators:
    """Calculate technical indicators"""
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> float:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        loss = loss.replace(0, 0.0001)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    @staticmethod
    def calculate_macd(data: pd.Series) -> Tuple[float, float, float]:
        exp1 = data.ewm(span=12, adjust=False).mean()
        exp2 = data.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd.iloc[-1], signal.iloc[-1], histogram.iloc[-1]
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20) -> Tuple[float, float, float]:
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper.iloc[-1], sma.iloc[-1], lower.iloc[-1]
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> float:
        ema = data.ewm(span=period, adjust=False).mean()
        return ema.iloc[-1]
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> float:
        sma = data.rolling(window=period).mean()
        return sma.iloc[-1]
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        high_low = high - low
        high_close = abs(high - close.shift())
        low_close = abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        high_diff = high.diff()
        low_diff = -low.diff()
        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        atr = TechnicalIndicators.calculate_atr(high, low, close, period)
        atr = max(atr, 0.0001)
        pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)
        di_sum = pos_di + neg_di
        di_sum = di_sum.replace(0, 0.0001)
        dx = 100 * abs(pos_di - neg_di) / di_sum
        adx = dx.rolling(window=period).mean()
        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25.0


class TradingStrategies:
    """Technical Analysis Trading Strategies"""
    
    @staticmethod
    def trend_following(close: pd.Series, ema_fast: float, ema_slow: float, adx: float) -> Dict:
        """Trend Following Strategy"""
        price = close.iloc[-1]
        
        signal = "NEUTRAL"
        strength = 0
        reason = []
        
        # Strong trend required (ADX > 25)
        if adx > 25:
            if ema_fast > ema_slow and price > ema_fast:
                signal = "BUY"
                strength = min(3, int((adx - 25) / 10) + 1)
                reason.append(f"Strong uptrend (ADX {adx:.1f})")
                reason.append(f"Price above fast EMA")
            elif ema_fast < ema_slow and price < ema_fast:
                signal = "SELL"
                strength = min(3, int((adx - 25) / 10) + 1)
                reason.append(f"Strong downtrend (ADX {adx:.1f})")
                reason.append(f"Price below fast EMA")
        else:
            reason.append("No clear trend (ADX < 25)")
        
        return {
            'name': 'Trend Following',
            'signal': signal,
            'strength': strength,
            'reason': reason
        }
    
    @staticmethod
    def mean_reversion(price: float, bb_upper: float, bb_lower: float, rsi: float) -> Dict:
        """Mean Reversion Strategy"""
        signal = "NEUTRAL"
        strength = 0
        reason = []
        
        # Oversold + at lower BB
        if rsi < 30 and price <= bb_lower * 1.01:
            signal = "BUY"
            strength = 3 if rsi < 25 else 2
            reason.append(f"Oversold (RSI {rsi:.1f})")
            reason.append("Price at lower Bollinger Band")
        # Overbought + at upper BB
        elif rsi > 70 and price >= bb_upper * 0.99:
            signal = "SELL"
            strength = 3 if rsi > 75 else 2
            reason.append(f"Overbought (RSI {rsi:.1f})")
            reason.append("Price at upper Bollinger Band")
        else:
            reason.append("Price in normal range")
        
        return {
            'name': 'Mean Reversion',
            'signal': signal,
            'strength': strength,
            'reason': reason
        }
    
    @staticmethod
    def breakout(price: float, high_20: float, low_20: float, volume: pd.Series, adx: float) -> Dict:
        """Breakout Strategy"""
        signal = "NEUTRAL"
        strength = 0
        reason = []
        
        recent_vol = volume.iloc[-1]
        avg_vol = volume.iloc[-20:].mean()
        high_volume = recent_vol > avg_vol * 1.5
        
        # Breakout above 20-day high with volume
        if price >= high_20 * 0.999:
            signal = "BUY"
            strength = 3 if high_volume else 2
            reason.append("Breakout above 20-day high")
            if high_volume:
                reason.append(f"High volume confirmation ({recent_vol/avg_vol:.1f}x avg)")
        # Breakdown below 20-day low with volume
        elif price <= low_20 * 1.001:
            signal = "SELL"
            strength = 3 if high_volume else 2
            reason.append("Breakdown below 20-day low")
            if high_volume:
                reason.append(f"High volume confirmation ({recent_vol/avg_vol:.1f}x avg)")
        else:
            reason.append("No breakout detected")
        
        return {
            'name': 'Breakout',
            'signal': signal,
            'strength': strength,
            'reason': reason
        }
    
    @staticmethod
    def momentum(macd: float, macd_signal: float, rsi: float, price_change_10d: float) -> Dict:
        """Momentum Strategy"""
        signal = "NEUTRAL"
        strength = 0
        reason = []
        
        bullish_signals = 0
        bearish_signals = 0
        
        # MACD bullish crossover
        if macd > macd_signal and macd > 0:
            bullish_signals += 1
            reason.append("MACD bullish crossover")
        elif macd < macd_signal and macd < 0:
            bearish_signals += 1
            reason.append("MACD bearish crossover")
        
        # RSI momentum
        if 50 < rsi < 70:
            bullish_signals += 1
            reason.append(f"Bullish RSI momentum ({rsi:.1f})")
        elif 30 < rsi < 50:
            bearish_signals += 1
            reason.append(f"Bearish RSI momentum ({rsi:.1f})")
        
        # Price momentum
        if price_change_10d > 2:
            bullish_signals += 1
            reason.append(f"Strong price momentum (+{price_change_10d:.1f}%)")
        elif price_change_10d < -2:
            bearish_signals += 1
            reason.append(f"Weak price momentum ({price_change_10d:.1f}%)")
        
        if bullish_signals >= 2:
            signal = "BUY"
            strength = bullish_signals
        elif bearish_signals >= 2:
            signal = "SELL"
            strength = bearish_signals
        else:
            reason = ["Mixed momentum signals"]
        
        return {
            'name': 'Momentum',
            'signal': signal,
            'strength': strength,
            'reason': reason
        }


class FuturesAnalyzer:
    """Main analyzer with live prices and strategies"""
    
    def __init__(self, av_key: str):
        self.fetcher = LiveDataFetcher(av_key)
        
    def analyze(self, futures_symbol: str) -> Optional[Dict]:
        """Analyze futures contract with live data"""
        
        # Get futures info
        futures_info = FuturesConverter.get_futures_info(futures_symbol)
        if not futures_info:
            print(f"❌ Unknown symbol: {futures_symbol}")
            print(f"Available: {', '.join(FuturesConverter.list_available_symbols())}")
            return None
        
        proxy_symbol = futures_info['proxy']
        
        print(f"\n{'='*70}")
        print(f"🔍 Analyzing {futures_info['name']} ({futures_symbol})")
        print(f"   Proxy ETF: {proxy_symbol}")
        print(f"{'='*70}")
        
        # Get live quote
        print(f"   📡 Fetching LIVE price from {proxy_symbol}...")
        live_quote = self.fetcher.get_live_quote(proxy_symbol)
        
        if not live_quote:
            print("❌ Could not fetch live price!")
            return None
        
        # Convert to futures price
        etf_price = live_quote['price']
        futures_price = FuturesConverter.convert_price(etf_price, futures_symbol)
        
        # Get historical data
        print(f"   📊 Fetching historical data...")
        historical = self.fetcher.get_historical_data(proxy_symbol, 'compact')
        
        if historical is None or len(historical) < 50:
            print("❌ Could not fetch historical data!")
            return None
        
        # Calculate indicators
        close = historical['Close']
        high = historical['High']
        low = historical['Low']
        volume = historical['Volume']
        
        rsi = TechnicalIndicators.calculate_rsi(close)
        macd, macd_signal, macd_hist = TechnicalIndicators.calculate_macd(close)
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(close)
        ema_9 = TechnicalIndicators.calculate_ema(close, 9)
        ema_21 = TechnicalIndicators.calculate_ema(close, 21)
        ema_50 = TechnicalIndicators.calculate_ema(close, 50)
        sma_200 = TechnicalIndicators.calculate_sma(close, 200)
        atr = TechnicalIndicators.calculate_atr(high, low, close)
        adx = TechnicalIndicators.calculate_adx(high, low, close)
        
        # 20-day highs/lows for breakout
        high_20 = high.iloc[-20:].max()
        low_20 = low.iloc[-20:].min()
        
        # Price change
        price_10d_ago = close.iloc[-10]
        price_change_10d = ((etf_price - price_10d_ago) / price_10d_ago) * 100
        
        # Run strategies
        strategies = []
        strategies.append(TradingStrategies.trend_following(close, ema_9, ema_21, adx))
        strategies.append(TradingStrategies.mean_reversion(etf_price, bb_upper, bb_lower, rsi))
        strategies.append(TradingStrategies.breakout(etf_price, high_20, low_20, volume, adx))
        strategies.append(TradingStrategies.momentum(macd, macd_signal, rsi, price_change_10d))
        
        # Calculate overall signal
        buy_score = sum(s['strength'] for s in strategies if s['signal'] == 'BUY')
        sell_score = sum(s['strength'] for s in strategies if s['signal'] == 'SELL')
        
        if buy_score >= sell_score + 2:
            overall_signal = "STRONG BUY 🚀"
        elif buy_score > sell_score:
            overall_signal = "BUY 📈"
        elif sell_score >= buy_score + 2:
            overall_signal = "STRONG SELL 📉"
        elif sell_score > buy_score:
            overall_signal = "SELL ⬇️"
        else:
            overall_signal = "HOLD ⏸️"
        
        # Calculate TP/SL
        atr_futures = atr * futures_info['multiplier']
        tick_size = futures_info['tick_size']
        
        def round_to_tick(value):
            return round(value / tick_size) * tick_size
        
        tp_long = round_to_tick(futures_price + (atr_futures * 2))
        sl_long = round_to_tick(futures_price - (atr_futures * 1))
        tp_short = round_to_tick(futures_price - (atr_futures * 2))
        sl_short = round_to_tick(futures_price + (atr_futures * 1))
        
        tp_long_ticks = (tp_long - futures_price) / tick_size
        sl_long_ticks = (futures_price - sl_long) / tick_size
        
        # Generate AI prediction
        prediction = self._generate_prediction(
            overall_signal, strategies, rsi, adx, 
            macd > macd_signal, etf_price > ema_21
        )
        
        return {
            'symbol': futures_symbol,
            'futures_info': futures_info,
            'live_data': {
                'etf_symbol': proxy_symbol,
                'etf_price': etf_price,
                'futures_price': futures_price,
                'change': live_quote['change'],
                'change_pct': live_quote['change_pct'],
                'volume': live_quote['volume'],
                'timestamp': live_quote['latest_day'],
                'open': live_quote['open'],
                'high': live_quote['high'],
                'low': live_quote['low']
            },
            'indicators': {
                'RSI': round(rsi, 2),
                'MACD': round(macd, 4),
                'MACD_Signal': round(macd_signal, 4),
                'BB_Upper': round(bb_upper, 2),
                'BB_Middle': round(bb_middle, 2),
                'BB_Lower': round(bb_lower, 2),
                'EMA_9': round(ema_9, 2),
                'EMA_21': round(ema_21, 2),
                'EMA_50': round(ema_50, 2),
                'SMA_200': round(sma_200, 2),
                'ATR': round(atr, 2),
                'ADX': round(adx, 2)
            },
            'strategies': strategies,
            'overall_signal': overall_signal,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'tp_sl': {
                'long': {
                    'tp': tp_long,
                    'sl': sl_long,
                    'tp_ticks': int(tp_long_ticks),
                    'sl_ticks': int(sl_long_ticks)
                },
                'short': {
                    'tp': tp_short,
                    'sl': sl_short
                }
            },
            'prediction': prediction
        }
    
    def _generate_prediction(self, signal: str, strategies: List[Dict], 
                            rsi: float, adx: float, macd_bullish: bool, 
                            above_ema21: bool) -> str:
        """Generate AI-style prediction"""
        
        # Count strategy agreements
        buy_strategies = [s['name'] for s in strategies if s['signal'] == 'BUY']
        sell_strategies = [s['name'] for s in strategies if s['signal'] == 'SELL']
        
        predictions = []
        
        if "STRONG BUY" in signal:
            predictions.append(f"Multiple strategies ({', '.join(buy_strategies)}) signal strong buying pressure.")
            if adx > 30:
                predictions.append("Strong trend strength suggests the upward momentum is likely to continue in the near term.")
            else:
                predictions.append("However, weak trend strength indicates caution - this move may not sustain without volume confirmation.")
        
        elif "BUY" in signal:
            predictions.append(f"Technical setup favors buyers with {', '.join(buy_strategies)} showing bullish signals.")
            if rsi < 50:
                predictions.append("RSI still has room to run higher before reaching overbought conditions, supporting further upside.")
            else:
                predictions.append("RSI approaching overbought territory suggests taking profits near resistance levels may be prudent.")
        
        elif "STRONG SELL" in signal:
            predictions.append(f"Multiple strategies ({', '.join(sell_strategies)}) indicate strong selling pressure ahead.")
            if adx > 30:
                predictions.append("Strong downtrend momentum suggests further downside is probable in the immediate future.")
            else:
                predictions.append("However, weak trend may lead to choppy price action rather than sustained decline.")
        
        elif "SELL" in signal:
            predictions.append(f"Bearish bias from {', '.join(sell_strategies)} suggests downside risk.")
            if rsi > 50:
                predictions.append("Elevated RSI levels support the case for mean reversion lower in coming sessions.")
            else:
                predictions.append("RSI already oversold may limit downside - watch for bounce signals.")
        
        else:  # HOLD
            predictions.append("Conflicting signals across strategies suggest sideways consolidation is most likely.")
            if adx < 25:
                predictions.append("Low ADX confirms range-bound conditions - best to wait for clearer directional move before entering.")
            else:
                predictions.append("Despite trend strength, lack of strategy consensus argues for patience until setup improves.")
        
        return " ".join(predictions)
    
    def display_results(self, results: Dict):
        """Display analysis results"""
        
        live = results['live_data']
        info = results['futures_info']
        indicators = results['indicators']
        
        print(f"\n{'='*70}")
        print(f"📊 LIVE MARKET DATA")
        print(f"{'='*70}")
        print(f"  Futures Price:    ${live['futures_price']:>10,.2f}")
        print(f"  Proxy ({live['etf_symbol']}):    ${live['etf_price']:>10,.2f}")
        print(f"  Change:           ${live['change']:>10,.2f} ({live['change_pct']}%)")
        print(f"  Today's Range:    ${live['low']:,.2f} - ${live['high']:,.2f}")
        print(f"  Volume:           {live['volume']:>10,}")
        print(f"  Last Updated:     {live['timestamp']}")
        
        print(f"\n{'='*70}")
        print(f"📋 CONTRACT SPECIFICATIONS")
        print(f"{'='*70}")
        print(f"  Contract:         {info['name']}")
        print(f"  Tick Size:        ${info['tick_size']}")
        print(f"  Tick Value:       ${info['tick_value']:.2f}")
        print(f"  Margin (approx):  ${info['margin']:,}")
        print(f"  Exchange:         {info['exchange']}")
        print(f"  Trading Hours:    {info['hours']}")
        
        print(f"\n{'='*70}")
        print(f"🎯 OVERALL SIGNAL: {results['overall_signal']}")
        print(f"📊 Score: BUY {results['buy_score']} | SELL {results['sell_score']}")
        print(f"{'='*70}")
        
        print(f"\n📈 TECHNICAL INDICATORS:")
        print(f"{'-'*70}")
        print(f"  RSI (14):         {indicators['RSI']:>8.2f}  {'(Oversold)' if indicators['RSI'] < 30 else '(Overbought)' if indicators['RSI'] > 70 else '(Neutral)'}")
        print(f"  MACD:             {indicators['MACD']:>8.4f}")
        print(f"  MACD Signal:      {indicators['MACD_Signal']:>8.4f}")
        print(f"\n  Bollinger Bands:")
        print(f"    Upper:          ${indicators['BB_Upper']:>8,.2f}")
        print(f"    Middle:         ${indicators['BB_Middle']:>8,.2f}")
        print(f"    Lower:          ${indicators['BB_Lower']:>8,.2f}")
        print(f"\n  Moving Averages:")
        print(f"    EMA 9:          ${indicators['EMA_9']:>8,.2f}")
        print(f"    EMA 21:         ${indicators['EMA_21']:>8,.2f}")
        print(f"    EMA 50:         ${indicators['EMA_50']:>8,.2f}")
        print(f"    SMA 200:        ${indicators['SMA_200']:>8,.2f}")
        print(f"\n  Volatility:")
        print(f"    ATR (14):       {indicators['ATR']:>8.2f}")
        print(f"    ADX (14):       {indicators['ADX']:>8.2f}")
        
        print(f"\n🎯 TRADING STRATEGIES ANALYSIS:")
        print(f"{'-'*70}")
        for strategy in results['strategies']:
            icon = "🟢" if strategy['signal'] == "BUY" else "🔴" if strategy['signal'] == "SELL" else "⚪"
            strength_bars = "★" * strategy['strength'] + "☆" * (3 - strategy['strength'])
            print(f"\n  {icon} {strategy['name']}: {strategy['signal']} {strength_bars}")
            for reason in strategy['reason']:
                print(f"     • {reason}")
        
        print(f"\n💰 RECOMMENDED LEVELS:")
        print(f"{'-'*70}")
        tp_sl = results['tp_sl']
        print(f"\n  LONG Position:")
        print(f"    Entry:        ${live['futures_price']:,.2f}")
        print(f"    Take Profit:  ${tp_sl['long']['tp']:,.2f} ({tp_sl['long']['tp_ticks']} ticks)")
        print(f"    Stop Loss:    ${tp_sl['long']['sl']:,.2f} ({tp_sl['long']['sl_ticks']} ticks)")
        print(f"    Risk/Reward:  2:1")
        
        print(f"\n  SHORT Position:")
        print(f"    Entry:        ${live['futures_price']:,.2f}")
        print(f"    Take Profit:  ${tp_sl['short']['tp']:,.2f}")
        print(f"    Stop Loss:    ${tp_sl['short']['sl']:,.2f}")
        print(f"    Risk/Reward:  2:1")
        
        print(f"\n{'='*70}")
        print(f"🤖 AI MARKET PREDICTION:")
        print(f"{'='*70}")
        print(f"\n{results['prediction']}")
        
        print(f"\n{'='*70}")
        print(f"⚠️  DISCLAIMER: Live data analysis for educational purposes only.")
        print(f"   Futures trading involves substantial risk of loss.")
        print(f"{'='*70}\n")


def main():
    """Main interactive function"""
    
    print("\n" + "="*70)
    print("🚀 LIVE FUTURES TRADING ANALYZER")
    print("="*70)
    print("\nReal-time analysis with technical strategies")
    print("\n" + "="*70)
    
    # Check API key
    if ALPHA_VANTAGE_KEY == "demo":
        print("\n⚠️  WARNING: Using demo API key (limited data)")
        print("Get your FREE key at: https://www.alphavantage.co/support/#api-key")
        print("Edit line 24 in this file to add your key\n")
    
    # Show available symbols
    available = FuturesConverter.list_available_symbols()
    print(f"\n📋 Available Futures Contracts:")
    print(f"{'-'*70}")
    for symbol in available:
        info = FuturesConverter.get_futures_info(symbol)
        print(f"  {symbol:<6} - {info['name']}")
    
    # Get user input
    print(f"\n{'-'*70}")
    user_input = input("Enter symbol (e.g., MNQ, NQ, GC, MGC): ").strip().upper()
    
    if not user_input:
        print("❌ No symbol entered. Exiting.")
        return
    
    # Analyze
    analyzer = FuturesAnalyzer(ALPHA_VANTAGE_KEY)
    results = analyzer.analyze(user_input)
    
    if results:
        analyzer.display_results(results)
    else:
        print("❌ Analysis failed. Please try again.")


if __name__ == '__main__':
    main()

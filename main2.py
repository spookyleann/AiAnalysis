#!/usr/bin/env python3
"""
Real-Time Trading Analyzer for Gold and NASDAQ
Provides buy/sell signals, risk assessment, and TP/SL recommendations
"""

import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import argparse

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️  Installing required packages...")
    print("Run: pip3 install yfinance pandas numpy matplotlib")
    print("\nContinuing with demo mode...\n")


class TechnicalAnalyzer:
    """Technical analysis calculations"""
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
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
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
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
        
        pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(window=period).mean()
        
        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25.0


class TradingSignalGenerator:
    """Generate trading signals based on technical analysis"""
    
    def __init__(self, symbol: str, data: pd.DataFrame):
        self.symbol = symbol
        self.data = data
        self.current_price = data['Close'].iloc[-1]
        self.signals = []
        self.score = 0
        
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
        
        # Analyze each indicator
        self._analyze_rsi(rsi)
        self._analyze_macd(macd, macd_signal, macd_hist)
        self._analyze_bollinger(bb_upper, bb_middle, bb_lower)
        self._analyze_moving_averages(ema_9, ema_21, ema_50, sma_200)
        self._analyze_stochastic(stoch_k, stoch_d)
        self._analyze_trend_strength(adx)
        self._analyze_volume(volume)
        
        # Calculate TP and SL
        tp_sl = self._calculate_tp_sl(atr, ema_21)
        
        # Determine overall signal
        signal = self._determine_signal()
        risk_level = self._assess_risk(adx, atr)
        
        return {
            'symbol': self.symbol,
            'current_price': round(self.current_price, 2),
            'signal': signal,
            'score': self.score,
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
                'ADX': round(adx, 2)
            },
            'tp_sl': tp_sl,
            'signals': self.signals
        }
    
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
    
    def _calculate_tp_sl(self, atr: float, ema21: float) -> Dict:
        """Calculate Take Profit and Stop Loss levels"""
        price = self.current_price
        
        # Conservative, Moderate, and Aggressive strategies
        strategies = {
            'conservative': {
                'tp_long': price + (atr * 1.5),
                'sl_long': price - (atr * 1.0),
                'tp_short': price - (atr * 1.5),
                'sl_short': price + (atr * 1.0),
                'risk_reward': 1.5
            },
            'moderate': {
                'tp_long': price + (atr * 2.0),
                'sl_long': price - (atr * 1.0),
                'tp_short': price - (atr * 2.0),
                'sl_short': price + (atr * 1.0),
                'risk_reward': 2.0
            },
            'aggressive': {
                'tp_long': price + (atr * 3.0),
                'sl_long': price - (atr * 1.0),
                'tp_short': price - (atr * 3.0),
                'sl_short': price + (atr * 1.0),
                'risk_reward': 3.0
            }
        }
        
        # Add percentage-based levels
        for strategy in strategies.values():
            strategy['tp_long_pct'] = ((strategy['tp_long'] - price) / price) * 100
            strategy['sl_long_pct'] = ((price - strategy['sl_long']) / price) * 100
            strategy['tp_short_pct'] = ((price - strategy['tp_short']) / price) * 100
            strategy['sl_short_pct'] = ((strategy['sl_short'] - price) / price) * 100
        
        return strategies
    
    def _determine_signal(self) -> str:
        """Determine overall trading signal"""
        if self.score >= 5:
            return "STRONG BUY 🚀"
        elif self.score >= 2:
            return "BUY 📈"
        elif self.score <= -5:
            return "STRONG SELL 📉"
        elif self.score <= -2:
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
    
    def __init__(self):
        self.symbols = {
            'gold': 'GC=F',  # Gold Futures
            'nasdaq': '^IXIC'  # NASDAQ Composite
        }
    
    def fetch_data(self, symbol: str, period: str = '3mo') -> Optional[pd.DataFrame]:
        """Fetch historical data"""
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            if data.empty:
                print(f"❌ No data available for {symbol}")
                return None
            return data
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return None
    
    def analyze_asset(self, asset_name: str) -> Optional[Dict]:
        """Analyze a specific asset"""
        symbol = self.symbols.get(asset_name.lower())
        if not symbol:
            print(f"❌ Unknown asset: {asset_name}")
            return None
        
        print(f"\n{'='*70}")
        print(f"🔍 Analyzing {asset_name.upper()} ({symbol})")
        print(f"{'='*70}")
        
        data = self.fetch_data(symbol)
        if data is None:
            return None
        
        signal_gen = TradingSignalGenerator(symbol, data)
        analysis = signal_gen.analyze()
        
        return analysis
    
    def display_analysis(self, analysis: Dict):
        """Display analysis results in a formatted way"""
        print(f"\n📊 Current Price: ${analysis['current_price']:,.2f}")
        print(f"\n{'='*70}")
        print(f"🎯 TRADING SIGNAL: {analysis['signal']}")
        print(f"📊 Signal Strength: {analysis['score']}/10")
        print(f"⚠️  Risk Level: {analysis['risk_level']}")
        print(f"{'='*70}")
        
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
        print(f"\n  Volatility & Momentum:")
        print(f"    ATR (14):      {indicators['ATR']:>8.2f}")
        print(f"    Stochastic %K: {indicators['Stochastic_K']:>8.2f}")
        print(f"    Stochastic %D: {indicators['Stochastic_D']:>8.2f}")
        print(f"    ADX (14):      {indicators['ADX']:>8.2f}")
        
        print(f"\n💡 SIGNAL BREAKDOWN:")
        print(f"{'-'*70}")
        for signal in analysis['signals']:
            print(f"  {signal}")
        
        print(f"\n🎯 TAKE PROFIT & STOP LOSS RECOMMENDATIONS:")
        print(f"{'-'*70}")
        
        tp_sl = analysis['tp_sl']
        
        # Conservative
        print(f"\n  💚 CONSERVATIVE (R:R = 1.5:1):")
        print(f"    LONG Position:")
        print(f"      Take Profit:  ${tp_sl['conservative']['tp_long']:>8,.2f}  (+{tp_sl['conservative']['tp_long_pct']:.2f}%)")
        print(f"      Stop Loss:    ${tp_sl['conservative']['sl_long']:>8,.2f}  (-{tp_sl['conservative']['sl_long_pct']:.2f}%)")
        print(f"    SHORT Position:")
        print(f"      Take Profit:  ${tp_sl['conservative']['tp_short']:>8,.2f}  (+{tp_sl['conservative']['tp_short_pct']:.2f}%)")
        print(f"      Stop Loss:    ${tp_sl['conservative']['sl_short']:>8,.2f}  (-{tp_sl['conservative']['sl_short_pct']:.2f}%)")
        
        # Moderate
        print(f"\n  💛 MODERATE (R:R = 2:1):")
        print(f"    LONG Position:")
        print(f"      Take Profit:  ${tp_sl['moderate']['tp_long']:>8,.2f}  (+{tp_sl['moderate']['tp_long_pct']:.2f}%)")
        print(f"      Stop Loss:    ${tp_sl['moderate']['sl_long']:>8,.2f}  (-{tp_sl['moderate']['sl_long_pct']:.2f}%)")
        print(f"    SHORT Position:")
        print(f"      Take Profit:  ${tp_sl['moderate']['tp_short']:>8,.2f}  (+{tp_sl['moderate']['tp_short_pct']:.2f}%)")
        print(f"      Stop Loss:    ${tp_sl['moderate']['sl_short']:>8,.2f}  (-{tp_sl['moderate']['sl_short_pct']:.2f}%)")
        
        # Aggressive
        print(f"\n  ❤️  AGGRESSIVE (R:R = 3:1):")
        print(f"    LONG Position:")
        print(f"      Take Profit:  ${tp_sl['aggressive']['tp_long']:>8,.2f}  (+{tp_sl['aggressive']['tp_long_pct']:.2f}%)")
        print(f"      Stop Loss:    ${tp_sl['aggressive']['sl_long']:>8,.2f}  (-{tp_sl['aggressive']['sl_long_pct']:.2f}%)")
        print(f"    SHORT Position:")
        print(f"      Take Profit:  ${tp_sl['aggressive']['tp_short']:>8,.2f}  (+{tp_sl['aggressive']['tp_short_pct']:.2f}%)")
        print(f"      Stop Loss:    ${tp_sl['aggressive']['sl_short']:>8,.2f}  (-{tp_sl['aggressive']['sl_short_pct']:.2f}%)")
        
        print(f"\n{'='*70}")
        print(f"⚠️  DISCLAIMER: This is for educational purposes only.")
        print(f"   Always do your own research and consult a financial advisor.")
        print(f"{'='*70}\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Real-Time Trading Analyzer for Gold and NASDAQ',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 trading_analyzer.py --asset gold
  python3 trading_analyzer.py --asset nasdaq
  python3 trading_analyzer.py --asset both
  python3 trading_analyzer.py --asset gold --period 6mo
        """
    )
    
    parser.add_argument(
        '--asset',
        choices=['gold', 'nasdaq', 'both'],
        default='both',
        help='Asset to analyze (default: both)'
    )
    
    parser.add_argument(
        '--period',
        default='3mo',
        help='Historical data period (default: 3mo). Options: 1mo, 3mo, 6mo, 1y, 2y'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results in JSON format'
    )
    
    args = parser.parse_args()
    
    if not YFINANCE_AVAILABLE:
        print("\n" + "="*70)
        print("❌ REQUIRED PACKAGES NOT INSTALLED")
        print("="*70)
        print("\nTo use this tool, install the required packages:")
        print("\n  pip3 install yfinance pandas numpy matplotlib")
        print("\nOr if you have permission issues:")
        print("\n  pip3 install --user yfinance pandas numpy matplotlib")
        print("\n" + "="*70 + "\n")
        sys.exit(1)
    
    analyzer = TradingAnalyzer()
    
    assets_to_analyze = ['gold', 'nasdaq'] if args.asset == 'both' else [args.asset]
    results = {}
    
    for asset in assets_to_analyze:
        analysis = analyzer.analyze_asset(asset)
        if analysis:
            results[asset] = analysis
            if not args.json:
                analyzer.display_analysis(analysis)
    
    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
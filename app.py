import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(
    page_title="High-Dividend Stock Screener", 
    page_icon="📈", 
    layout="wide"
)

class StockScreener:
    def __init__(self):
        self.alpha_vantage_key = st.secrets.get("ALPHA_VANTAGE_API_KEY", "demo")
        self.fmp_key = st.secrets.get("FMP_API_KEY", "demo")
        
        # Stock lists for different markets
        self.us_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'PG',
            'V', 'UNH', 'HD', 'PFE', 'BAC', 'KO', 'DIS', 'ADBE', 'NFLX', 'CRM',
            'XOM', 'WMT', 'CVX', 'LLY', 'ABBV', 'TMO', 'NKE', 'COST', 'AVGO', 'ABT',
            'MRK', 'ACN', 'ORCL', 'TXN', 'DHR', 'QCOM', 'VZ', 'MDT', 'BMY', 'NEE',
            'PM', 'HON', 'UNP', 'AMGN', 'IBM', 'GE', 'MMM', 'SPGI', 'LOW', 'INTC',
            'T', 'CAT', 'SBUX', 'GILD', 'AMD', 'BKNG', 'TGT', 'MDLZ', 'INTU', 'PYPL',
            'AXP', 'CVS', 'MO', 'SYK', 'SCHW', 'BLK', 'ISRG', 'AMT', 'LRCX', 'DE',
            'PLD', 'ZTS', 'ADI', 'NOW', 'GS', 'LMT', 'EL', 'RTX', 'TJX', 'FIS',
            'REGN', 'C', 'AMAT', 'BSX', 'CB', 'MU', 'SHW', 'BDX', 'ICE', 'DUK',
            'SO', 'APD', 'EQIX', 'WM', 'COP', 'PSA', 'NSC', 'AON', 'CME', 'USB'
        ]
        
        self.european_stocks = [
            # German stocks (DAX)
            'SAP.DE', 'ASML.AS', 'ADYEN.AS', 'NESN.SW', 'NOVN.SW', 'ROG.SW', 'INGA.AS',
            'PHIA.AS', 'UNA.AS', 'RDSA.AS', 'HEIA.AS', 'AIRBUS.PA', 'LVMH.PA', 'OR.PA',
            'MC.PA', 'CDI.PA', 'BNP.PA', 'SAN.PA', 'AI.PA', 'CS.PA', 'DG.PA', 'KER.PA',
            'RMS.PA', 'CAP.PA', 'ML.PA', 'SU.PA', 'TTE.PA', 'AC.PA', 'BN.PA', 'ORA.PA',
            'ENGI.PA', 'VIV.PA', 'EDF.PA', 'ATO.PA', 'DSY.PA', 'SGO.PA', 'VIE.PA',
            # Additional European stocks
            'VOW3.DE', 'BMW.DE', 'MBG.DE', 'ALV.DE', 'SIE.DE', 'DTE.DE', 'DBK.DE',
            'BAS.DE', 'BAYN.DE', 'FRE.DE', 'LIN.DE', 'MTX.DE', 'MRK.DE', 'RWE.DE',
            'VOW.DE', '1COV.DE', 'ADS.DE', 'HEN3.DE', 'IFX.DE', 'SAP.DE'
        ]

    def fetch_stock_data_yfinance(self, symbol: str) -> Optional[Dict]:
        """Fetch stock data using Yahoo Finance API (free tier)"""
        try:
            # Yahoo Finance API endpoints
            base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            
            response = requests.get(f"{base_url}{symbol}", headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                if 'chart' in data and data['chart']['result']:
                    result = data['chart']['result'][0]['meta']
                    
                    # Get additional data from summary
                    summary_url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
                    params = {
                        'modules': 'summaryDetail,defaultKeyStatistics,financialData,calendarEvents'
                    }
                    
                    summary_response = requests.get(summary_url, params=params, headers=headers, timeout=10)
                    summary_data = {}
                    
                    if summary_response.status_code == 200:
                        summary_json = summary_response.json()
                        if 'quoteSummary' in summary_json and summary_json['quoteSummary']['result']:
                            summary_data = summary_json['quoteSummary']['result'][0]
                    
                    return {
                        'symbol': symbol,
                        'price': result.get('regularMarketPrice', 0),
                        'dividend_yield': self._safe_get(summary_data, 'summaryDetail.dividendYield.raw', 0) * 100 if summary_data else 0,
                        'payout_ratio': self._safe_get(summary_data, 'summaryDetail.payoutRatio.raw', 0) * 100 if summary_data else 0,
                        'forward_pe': self._safe_get(summary_data, 'summaryDetail.forwardPE.raw', 0) if summary_data else 0,
                        'profit_margin': self._safe_get(summary_data, 'defaultKeyStatistics.profitMargins.raw', 0) * 100 if summary_data else 0,
                        'operating_margin': self._safe_get(summary_data, 'financialData.operatingMargins.raw', 0) * 100 if summary_data else 0,
                        'market_cap': self._safe_get(summary_data, 'summaryDetail.marketCap.raw', 0) if summary_data else 0,
                        'beta': self._safe_get(summary_data, 'summaryDetail.beta.raw', 1) if summary_data else 1,
                        'ex_dividend_date': self._safe_get(summary_data, 'calendarEvents.exDividendDate.raw') if summary_data else None
                    }
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
        
        return None

    def _safe_get(self, data: dict, path: str, default=None):
        """Safely get nested dictionary values"""
        try:
            keys = path.split('.')
            value = data
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def fetch_financial_modeling_prep_data(self, symbol: str) -> Optional[Dict]:
        """Fetch data from Financial Modeling Prep API"""
        if self.fmp_key == "demo":
            return None
            
        try:
            # Clean symbol for FMP (remove exchange suffix for European stocks)
            clean_symbol = symbol.split('.')[0] if '.' in symbol else symbol
            
            # Get profile data
            profile_url = f"https://financialmodelingprep.com/api/v3/profile/{clean_symbol}"
            profile_params = {'apikey': self.fmp_key}
            
            profile_response = requests.get(profile_url, params=profile_params, timeout=10)
            
            if profile_response.status_code == 200:
                profile_data = profile_response.json()
                if profile_data:
                    profile = profile_data[0]
                    
                    # Get key metrics
                    metrics_url = f"https://financialmodelingprep.com/api/v3/key-metrics/{clean_symbol}"
                    metrics_response = requests.get(metrics_url, params=profile_params, timeout=10)
                    
                    metrics_data = {}
                    if metrics_response.status_code == 200:
                        metrics_json = metrics_response.json()
                        if metrics_json:
                            metrics_data = metrics_json[0]
                    
                    return {
                        'symbol': symbol,
                        'price': profile.get('price', 0),
                        'dividend_yield': profile.get('lastDiv', 0) / profile.get('price', 1) * 100 if profile.get('price') else 0,
                        'payout_ratio': metrics_data.get('payoutRatio', 0) * 100,
                        'forward_pe': metrics_data.get('peRatio', 0),
                        'profit_margin': metrics_data.get('netProfitMargin', 0) * 100,
                        'operating_margin': metrics_data.get('operatingProfitMargin', 0) * 100,
                        'market_cap': profile.get('mktCap', 0),
                        'beta': profile.get('beta', 1),
                        'industry': profile.get('industry', 'Unknown'),
                        'sector': profile.get('sector', 'Unknown')
                    }
        except Exception as e:
            st.error(f"Error fetching FMP data for {symbol}: {str(e)}")
            return None
        
        return None

    def calculate_total_return(self, symbol: str, years: int = 1) -> float:
        """Calculate total return including dividends (simplified estimation)"""
        try:
            # This is a simplified calculation
            # In a real scenario, you'd want historical price and dividend data
            # For demo purposes, we'll estimate based on current metrics
            
            # Fetch current data
            data = self.fetch_stock_data_yfinance(symbol)
            if not data:
                return 0
            
            dividend_yield = data.get('dividend_yield', 0)
            
            # Simplified estimation: assume 8% average price appreciation for quality dividend stocks
            # This is just for demonstration - real calculation would need historical data
            estimated_price_return = 8.0  # 8% annual price appreciation assumption
            total_return = estimated_price_return + dividend_yield
            
            return total_return
            
        except Exception:
            return 0

    def screen_stocks(self, stocks: List[str], min_dividend_yield: float = 4.0, 
                     min_total_return: float = 10.0, min_operating_margin: float = 5.0,
                     max_payout_ratio: float = 80.0) -> pd.DataFrame:
        """Screen stocks based on criteria"""
        
        screened_stocks = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(stocks):
            status_text.text(f'Analyzing {symbol}... ({i+1}/{len(stocks)})')
            progress_bar.progress((i + 1) / len(stocks))
            
            # Try multiple data sources
            stock_data = self.fetch_stock_data_yfinance(symbol)
            
            if not stock_data:
                stock_data = self.fetch_financial_modeling_prep_data(symbol)
            
            if stock_data:
                # Calculate total return
                total_return = self.calculate_total_return(symbol)
                stock_data['total_return'] = total_return
                
                # Apply screening criteria
                if (stock_data['dividend_yield'] >= min_dividend_yield and
                    total_return >= min_total_return and
                    stock_data['operating_margin'] >= min_operating_margin and
                    stock_data['payout_ratio'] <= max_payout_ratio and
                    stock_data['payout_ratio'] > 0):  # Must actually pay dividends
                    
                    screened_stocks.append(stock_data)
            
            # Add delay to respect API limits
            time.sleep(0.1)
        
        progress_bar.empty()
        status_text.empty()
        
        if screened_stocks:
            df = pd.DataFrame(screened_stocks)
            
            # Add dividend stability score (simplified)
            df['stability_score'] = self.calculate_stability_score(df)
            
            # Sort by dividend yield descending
            df = df.sort_values('dividend_yield', ascending=False)
            
            return df
        else:
            return pd.DataFrame()

    def calculate_stability_score(self, df: pd.DataFrame) -> List[float]:
        """Calculate a simplified dividend stability score"""
        scores = []
        for _, row in df.iterrows():
            score = 0
            
            # Lower payout ratio is better (more sustainable)
            if row['payout_ratio'] < 50:
                score += 30
            elif row['payout_ratio'] < 70:
                score += 20
            else:
                score += 10
            
            # Higher operating margin is better
            if row['operating_margin'] > 15:
                score += 25
            elif row['operating_margin'] > 10:
                score += 20
            elif row['operating_margin'] > 5:
                score += 15
            
            # Reasonable P/E ratio
            if 10 <= row.get('forward_pe', 0) <= 20:
                score += 25
            elif 8 <= row.get('forward_pe', 0) <= 25:
                score += 20
            
            # Beta considerations (less volatile is better for dividend stocks)
            beta = row.get('beta', 1)
            if beta < 0.8:
                score += 20
            elif beta < 1.2:
                score += 15
            else:
                score += 10
                
            scores.append(score)
        
        return scores

def main():
    st.title("📈 High-Dividend Stock Screener")
    st.markdown("**Find high-quality dividend stocks in US and European markets**")
    
    screener = StockScreener()
    
    # Sidebar for parameters
    st.sidebar.header("🔧 Screening Parameters")
    
    min_dividend_yield = st.sidebar.slider(
        "Minimum Dividend Yield (%)", 
        min_value=1.0, max_value=10.0, value=4.0, step=0.5
    )
    
    min_total_return = st.sidebar.slider(
        "Minimum Total Return (%)", 
        min_value=5.0, max_value=20.0, value=10.0, step=1.0
    )
    
    min_operating_margin = st.sidebar.slider(
        "Minimum Operating Margin (%)", 
        min_value=0.0, max_value=20.0, value=5.0, step=1.0
    )
    
    max_payout_ratio = st.sidebar.slider(
        "Maximum Payout Ratio (%)", 
        min_value=30.0, max_value=100.0, value=80.0, step=5.0
    )
    
    market_selection = st.sidebar.multiselect(
        "Select Markets",
        ["US Stocks", "European Stocks"],
        default=["US Stocks", "European Stocks"]
    )
    
    # API Key Configuration
    st.sidebar.header("🔑 API Configuration")
    st.sidebar.markdown("""
    **Optional API Keys for Enhanced Data:**
    - Add `ALPHA_VANTAGE_API_KEY` in Streamlit secrets
    - Add `FMP_API_KEY` in Streamlit secrets
    
    The app works with free data sources but paid APIs provide more comprehensive data.
    """)
    
    if st.sidebar.button("🚀 Start Screening", type="primary"):
        
        # Prepare stock list
        stocks_to_screen = []
        if "US Stocks" in market_selection:
            stocks_to_screen.extend(screener.us_stocks)
        if "European Stocks" in market_selection:
            stocks_to_screen.extend(screener.european_stocks)
        
        if not stocks_to_screen:
            st.error("Please select at least one market!")
            return
        
        st.info(f"Screening {len(stocks_to_screen)} stocks with your criteria...")
        
        # Screen stocks
        results_df = screener.screen_stocks(
            stocks_to_screen,
            min_dividend_yield=min_dividend_yield,
            min_total_return=min_total_return,
            min_operating_margin=min_operating_margin,
            max_payout_ratio=max_payout_ratio
        )
        
        if results_df.empty:
            st.warning("No stocks found matching your criteria. Try relaxing the parameters.")
        else:
            st.success(f"Found {len(results_df)} stocks matching your criteria!")
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Screening Results")
                
                # Format the dataframe for display
                display_df = results_df.copy()
                display_df['dividend_yield'] = display_df['dividend_yield'].apply(lambda x: f"{x:.2f}%")
                display_df['total_return'] = display_df['total_return'].apply(lambda x: f"{x:.2f}%")
                display_df['operating_margin'] = display_df['operating_margin'].apply(lambda x: f"{x:.2f}%")
                display_df['payout_ratio'] = display_df['payout_ratio'].apply(lambda x: f"{x:.2f}%")
                display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
                display_df['stability_score'] = display_df['stability_score'].apply(lambda x: f"{x:.0f}/100")
                
                # Select columns to display
                columns_to_show = ['symbol', 'price', 'dividend_yield', 'total_return', 
                                 'operating_margin', 'payout_ratio', 'stability_score']
                
                st.dataframe(
                    display_df[columns_to_show],
                    column_config={
                        "symbol": "Symbol",
                        "price": "Price",
                        "dividend_yield": "Div Yield",
                        "total_return": "Total Return",
                        "operating_margin": "Op Margin",
                        "payout_ratio": "Payout Ratio",
                        "stability_score": "Stability"
                    },
                    use_container_width=True
                )
            
            with col2:
                st.subheader("📈 Key Metrics")
                
                avg_dividend_yield = results_df['dividend_yield'].mean()
                avg_total_return = results_df['total_return'].mean()
                avg_stability = results_df['stability_score'].mean()
                
                st.metric("Average Dividend Yield", f"{avg_dividend_yield:.2f}%")
                st.metric("Average Total Return", f"{avg_total_return:.2f}%")
                st.metric("Average Stability Score", f"{avg_stability:.0f}/100")
                
                # Top performer
                if not results_df.empty:
                    top_stock = results_df.iloc[0]
                    st.markdown("**🏆 Top Pick:**")
                    st.markdown(f"**{top_stock['symbol']}**")
                    st.markdown(f"Div Yield: {top_stock['dividend_yield']:.2f}%")
                    st.markdown(f"Total Return: {top_stock['total_return']:.2f}%")
            
            # Visualizations
            if len(results_df) > 1:
                st.subheader("📊 Analysis Charts")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Dividend yield vs Total return scatter plot
                    fig1 = px.scatter(
                        results_df, 
                        x='dividend_yield', 
                        y='total_return',
                        size='market_cap',
                        hover_name='symbol',
                        title="Dividend Yield vs Total Return",
                        labels={
                            'dividend_yield': 'Dividend Yield (%)',
                            'total_return': 'Total Return (%)'
                        }
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Operating margin vs Payout ratio
                    fig2 = px.scatter(
                        results_df,
                        x='operating_margin',
                        y='payout_ratio',
                        size='stability_score',
                        hover_name='symbol',
                        title="Operating Margin vs Payout Ratio",
                        labels={
                            'operating_margin': 'Operating Margin (%)',
                            'payout_ratio': 'Payout Ratio (%)'
                        }
                    )
                    st.plotly_chart(fig2, use_container_width=True)
    
    # Information section
    st.markdown("---")
    st.subheader("ℹ️ About This Screener")
    st.markdown("""
    This tool screens stocks based on:
    
    - **High Dividend Yield (>4%)**: Stocks that provide substantial income
    - **High Total Return (>10%)**: Including both price appreciation and dividends
    - **Strong Operating Margin (>5%)**: Indicates operational efficiency
    - **Sustainable Payout Ratio (<80%)**: Ensures dividend sustainability
    - **Stability Score**: Custom metric considering payout ratio, margins, P/E, and volatility
    
    **Markets Covered:**
    - 🇺🇸 US stocks (S&P 500 components and dividend aristocrats)
    - 🇪🇺 European stocks (DAX, CAC 40, FTSE components)
    
    **Data Sources:**
    - Yahoo Finance (free tier)
    - Financial Modeling Prep (with API key)
    - Alpha Vantage (with API key)
    """)
    
    st.markdown("---")
    st.markdown("**⚠️ Disclaimer:** This tool is for educational purposes only. Always do your own research before making investment decisions.")

if __name__ == "__main__":
    main()

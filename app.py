import streamlit as st
import pandas as pd
from data_ingestion import get_market_data, get_asset_info
from intelligence import generate_recommendation_gemini, generate_recommendation_ollama

st.set_page_config(page_title="AI Personal Investment Assistant", layout="wide")

st.title("📈 AI Personal Investment Assistant")
st.markdown("Your personal advisor for Stocks, Crypto, and Index Funds/ETFs.")

# Sidebar Configuration
st.sidebar.header("Configuration")
ai_model_choice = st.sidebar.selectbox("Select AI Model", ["Gemini API", "Local Ollama"])
if ai_model_choice == "Local Ollama":
    ollama_model = st.sidebar.text_input("Ollama Model Name", value="hermes")

st.sidebar.markdown("---")
st.sidebar.header("Asset Selection")
asset_type = st.sidebar.selectbox("Asset Type", ["Stocks", "Crypto", "Index Funds / ETFs"])

# Defaults based on asset type
if asset_type == "Stocks":
    default_ticker = "AAPL"
elif asset_type == "Crypto":
    default_ticker = "BTC-USD"
else:
    default_ticker = "SPY"

ticker_input = st.sidebar.text_input("Enter Ticker Symbol", value=default_ticker).upper()

# Use caching to avoid hitting rate limits (5 minute cache)
@st.cache_data(ttl=300)
def load_data(ticker):
    data = get_market_data(ticker, period="1y", interval="1d")
    info = get_asset_info(ticker)
    return data, info

if ticker_input:
    with st.spinner("Fetching data..."):
        market_data, asset_info = load_data(ticker_input)
        
    if not market_data.empty:
        st.subheader(f"Data Snapshot for {ticker_input}")
        
        # Display fundamentals
        if asset_info:
            with st.expander("Fundamental Information", expanded=True):
                cols = st.columns(4)
                cols[0].metric("Previous Close", asset_info.get("previousClose", "N/A"))
                
                # Format Market Cap if it's a number
                mcap = asset_info.get("marketCap", "N/A")
                if isinstance(mcap, (int, float)):
                    mcap = f"${mcap:,.0f}"
                cols[1].metric("Market Cap", mcap)
                
                cols[2].metric("Trailing P/E", asset_info.get("trailingPE", "N/A"))
                cols[3].metric("Dividend Yield", asset_info.get("dividendYield", "N/A"))
                
                st.write(f"**Sector**: {asset_info.get('sector', 'N/A')} | **Industry**: {asset_info.get('industry', 'N/A')}")
        
        # Display chart
        st.line_chart(market_data['Close'])
        
        st.markdown("---")
        st.subheader("🤖 AI Recommendation")
        st.markdown("Get a recommendation with foundational reasons and supportive information based on technicals and fundamentals.")
        
        query = st.text_input("Ask a specific question about this asset (optional):", placeholder="e.g. Is this a good time to buy given recent inflation reports?")
        
        if st.button("Generate Recommendation"):
            with st.spinner("Analyzing data and consulting AI..."):
                if ai_model_choice == "Gemini API":
                    rec = generate_recommendation_gemini(asset_info, market_data, query)
                else:
                    rec = generate_recommendation_ollama(asset_info, market_data, query, model=ollama_model)
                
                st.info("Analysis Complete")
                st.markdown(rec)
    else:
        st.error(f"Could not fetch data for {ticker_input}. Please check the symbol.")

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

# Define Top 10 lists
TOP_LISTS = {
    "US Stocks": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "BRK-B", "LLY", "AVGO", "TSLA"],
    "NZ Stocks": ["FPH.NZ", "SPK.NZ", "AIA.NZ", "CEN.NZ", "IFT.NZ", "MEL.NZ", "MCY.NZ", "EBO.NZ", "FBU.NZ", "POT.NZ"],
    "Crypto": ["BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "SOL-USD", "USDC-USD", "XRP-USD", "DOGE-USD", "TON11419-USD", "ADA-USD"],
    "US Index Funds": ["SPY", "IVV", "VOO", "QQQ", "VTI", "BND", "VEA", "VTV", "IEFA", "VUG"],
    "NZ Index Funds": ["FNZ.NZ", "MDZ.NZ", "OZY.NZ", "USF.NZ", "TWF.NZ", "DIV.NZ", "NPF.NZ", "TNZ.NZ", "EUF.NZ", "EMG.NZ"]
}

@st.cache_data(ttl=300)
def load_data(ticker):
    data = get_market_data(ticker, period="1y", interval="1d")
    info = get_asset_info(ticker)
    return data, info

# --- HOMEPAGE TOP 10 LISTS ---
st.markdown("---")
st.header("Top 10 Market Insights")
st.markdown("Select an asset below to immediately load its deep dive analysis.")
tabs = st.tabs(list(TOP_LISTS.keys()))

for i, (tab_name, tickers) in enumerate(TOP_LISTS.items()):
    with tabs[i]:
        cols = st.columns(5)
        for j, ticker in enumerate(tickers):
            # Using a button for each ticker. When clicked, it updates session state.
            if cols[j % 5].button(f"Analyze {ticker}", key=f"btn_{tab_name}_{ticker}"):
                st.session_state['selected_ticker'] = ticker

st.sidebar.markdown("---")
st.sidebar.header("Asset Selection")

# Retrieve selected ticker from session state, defaulting to AAPL
selected_ticker = st.session_state.get('selected_ticker', 'AAPL')
ticker_input = st.sidebar.text_input("Enter Ticker Symbol", value=selected_ticker).upper()

# Ensure the session state stays updated if user manually types
if ticker_input != st.session_state.get('selected_ticker'):
    st.session_state['selected_ticker'] = ticker_input

if ticker_input:
    st.markdown("---")
    st.header(f"Deep Dive: {ticker_input}")
    with st.spinner(f"Fetching complete data snapshot for {ticker_input}..."):
        market_data, asset_info = load_data(ticker_input)
        
    if not market_data.empty:
        # Display fundamentals
        if asset_info:
            with st.expander("Fundamental Information & Supportive Metrics", expanded=True):
                st.markdown("### Valuation & Price")
                c1, c2, c3, c4 = st.columns(4)
                
                def fmt(val, currency=False):
                    if val == "N/A" or val is None: return "N/A"
                    try:
                        v = float(val)
                        if currency:
                            return f"${v:,.2f}"
                        return f"{v:,.2f}"
                    except:
                        return str(val)

                def fmt_pct(val):
                    if val == "N/A" or val is None: return "N/A"
                    try:
                        return f"{float(val)*100:.2f}%"
                    except:
                        return str(val)

                c1.metric("Previous Close", fmt(asset_info.get("previousClose"), True))
                c2.metric("Market Cap", fmt(asset_info.get("marketCap"), True))
                c3.metric("Trailing P/E", fmt(asset_info.get("trailingPE")))
                c4.metric("Forward P/E", fmt(asset_info.get("forwardPE")))
                
                st.markdown("### Technical & Risk Profile")
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Beta (Volatility)", fmt(asset_info.get("beta")))
                c6.metric("50-Day Avg", fmt(asset_info.get("fiftyDayAverage"), True))
                c7.metric("200-Day Avg", fmt(asset_info.get("twoHundredDayAverage"), True))
                c8.metric("Short Ratio", fmt(asset_info.get("shortRatio")))
                
                st.markdown("### Financial Health & Growth")
                c9, c10, c11, c12 = st.columns(4)
                c9.metric("Dividend Yield", fmt_pct(asset_info.get("dividendYield")))
                c10.metric("Price to Book", fmt(asset_info.get("priceToBook")))
                c11.metric("Revenue Growth", fmt_pct(asset_info.get("revenueGrowth")))
                c12.metric("Debt to Equity", fmt(asset_info.get("debtToEquity")))
                
                st.write(f"**Sector**: {asset_info.get('sector', 'N/A')} | **Industry**: {asset_info.get('industry', 'N/A')}")
        
        # Display chart
        st.line_chart(market_data['Close'])
        
        st.markdown("---")
        st.subheader("🤖 AI Recommendation")
        st.markdown("Get an insight-driven recommendation based on recent technicals, fundamentals, and AI reasoning.")
        
        query = st.text_input("Ask a specific question about this asset (optional):", placeholder="e.g. Is this a good time to buy given recent inflation reports?", key="query_input")
        
        if st.button("Generate Recommendation", type="primary"):
            with st.spinner("Analyzing data and consulting AI..."):
                if ai_model_choice == "Gemini API":
                    rec = generate_recommendation_gemini(asset_info, market_data, query)
                else:
                    rec = generate_recommendation_ollama(asset_info, market_data, query, model=ollama_model)
                
                st.info("Analysis Complete")
                st.markdown(rec)
    else:
        st.error(f"Could not fetch data for {ticker_input}. Please check the symbol.")

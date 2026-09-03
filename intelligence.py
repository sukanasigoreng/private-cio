import pandas as pd
from google import genai
import os

def calculate_sma(data: pd.DataFrame, window: int = 20) -> pd.Series:
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = data['Close'].diff()
    
    # Calculate exponential moving average for gain and loss
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_recommendation_gemini(asset_info: dict, market_data: pd.DataFrame, query: str = None) -> str:
    """
    Generate recommendation using Gemini API.
    """
    try:
        client = genai.Client()
    except Exception as e:
        return "Could not initialize Gemini Client. Did you set GEMINI_API_KEY?"
    
    ta_summary = "Technical Analysis: Not enough data."
    if not market_data.empty and len(market_data) > 20:
        current_price = market_data['Close'].iloc[-1]
        sma_20 = calculate_sma(market_data, 20).iloc[-1]
        rsi_14 = calculate_rsi(market_data, 14).iloc[-1]
        
        ta_summary = f"Current Price: {current_price:.2f}\n20-Day SMA: {sma_20:.2f}\n14-Day RSI: {rsi_14:.2f}"

    prompt = f"""
You are a sharp, analytical Personal Investment Assistant. 
Provide a detailed recommendation on whether to buy, sell, or hold the following asset, along with foundational reasons.

Asset Info:
{asset_info}

Technical Indicators:
{ta_summary}
"""
    if query:
        prompt += f"\n\nUser Question: {query}"
        
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini API: {e}. Please ensure you have set the GEMINI_API_KEY environment variable."

def generate_recommendation_ollama(asset_info: dict, market_data: pd.DataFrame, query: str = None, model: str = "hermes") -> str:
    """
    Generate recommendation using local Ollama model (e.g., hermes).
    """
    import ollama
    
    ta_summary = "Technical Analysis: Not enough data."
    if not market_data.empty and len(market_data) > 20:
        current_price = market_data['Close'].iloc[-1]
        sma_20 = calculate_sma(market_data, 20).iloc[-1]
        rsi_14 = calculate_rsi(market_data, 14).iloc[-1]
        ta_summary = f"Current Price: {current_price:.2f}\n20-Day SMA: {sma_20:.2f}\n14-Day RSI: {rsi_14:.2f}"

    prompt = f"""
You are a sharp, analytical Personal Investment Assistant. 
Provide a detailed recommendation on whether to buy, sell, or hold the following asset, along with foundational reasons.

Asset Info:
{asset_info}

Technical Indicators:
{ta_summary}
"""
    if query:
        prompt += f"\n\nUser Question: {query}"

    try:
        response = ollama.chat(model=model, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        return response['message']['content']
    except Exception as e:
        return f"Error communicating with Ollama (model: {model}): {e}. Make sure Ollama is running and the model is pulled."

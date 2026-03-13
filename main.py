from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from database.database_service import process_results, store_agent_reports
from dotenv import load_dotenv
from datetime import datetime
import os

# Load environment variables from .env file
load_dotenv()

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["backend_url"]= "https://api.groq.com/openai/v1"
config["deep_think_llm"] = "meta-llama/llama-4-scout-17b-16e-instruct"  # Use a different model
config["quick_think_llm"] = "meta-llama/llama-4-scout-17b-16e-instruct"  # Use a different model
config["max_debate_rounds"] = 1  # Increase debate rounds

# Configure data vendors (default uses yfinance, no extra API keys needed)
config["data_vendors"] = {
    "core_stock_apis": "yfinance",           # Options: alpha_vantage, yfinance
    "technical_indicators": "yfinance",      # Options: alpha_vantage, yfinance
    "fundamental_data": "yfinance",          # Options: alpha_vantage, yfinance
    "news_data": "yfinance",                 # Options: alpha_vantage, yfinance
}

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)


company_symbol = str(os.environ.get("COMPANY_NAME"))
analysis_date = datetime.today().strftime('%Y-%m-%d')

# forward propagate
_, decision = ta.propagate(company_symbol + ".NS", analysis_date)

process_results(company_symbol, analysis_date, decision)
store_agent_reports()
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns

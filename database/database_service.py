import os
import psycopg2
from psycopg2.extras import RealDictCursor
import json 
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT", 5432),
            sslmode="require"
        )
        return conn
    except Exception as e:
        print("Database connection failed:")
        print(str(e))
        raise

def execute_query(query, params=None):
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            if cur.description:  # If the query returns data (e.g., SELECT)
                result = cur.fetchall()
            else:
                result = []
        conn.commit() # Explicitly commit the transaction
        return result
    except Exception as e:
        if conn:
            conn.rollback()
        print("Query execution failed:")
        print(str(e))
        raise
    finally:
        # CRITICAL: Always close the connection to flush network buffers
        if conn:
            conn.close()

def construct_result_path():
    base_path = "./eval_results"
    company_folder = os.listdir(base_path)[0]
    logs_path = os.path.join(base_path, company_folder, "TradingAgentsStrategy_logs")
    json_file = os.listdir(logs_path)[0]
    return os.path.join(logs_path, json_file)

def store_agent_reports():
    json_path = construct_result_path()
    print("Reading file:", json_path)

    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # JSON structure contains date as key
    data = list(raw_data.values())[0]

    company = data["company_of_interest"]
    analysis_date = data["trade_date"]

    agent_reports = {
        "market": data.get("market_report"),
        "sentiment": data.get("sentiment_report"),
        "news": data.get("news_report"),
        "fundamentals": data.get("fundamentals_report"),
        "research_manager": data.get("investment_plan"),
        "trader": data.get("trader_investment_decision"),
        "risk_manager": data.get("final_trade_decision")
    }

    query = """
    INSERT INTO agent_reports(company_name, report_date, agent_name, report)
    VALUES (%s, %s, %s, %s)
    """
    for agent, report in agent_reports.items():
        if report:
            # If 'report' is a dictionary, use json.dumps(report) instead
            report_data = json.dumps(report) if isinstance(report, dict) else report
            execute_query(query, (company, analysis_date, agent, report_data)) 

def process_results(company_symbol, analysis_date, final_decision):
    query = """
    INSERT INTO final_trading_decisions(company_symbol, trade_date, decision)
    VALUES (%s, %s, %s)
    """
    
    # Ensure complex objects are converted to strings/JSON for the database
    if isinstance(final_decision, dict) or isinstance(final_decision, list):
        final_decision = json.dumps(final_decision)
    else:
        final_decision = str(final_decision)

    execute_query(query, (company_symbol, analysis_date, final_decision))
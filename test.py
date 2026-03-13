# import json 
# import os
# from database.database_service import execute_query

# # from tradingagents.dataflows.interface import get_yfinance_income_statement

# # result = get_yfinance_income_statement("TCS.NS")
# # print("Income statement for TCS.NS:", result)
# # if not result or result.strip() == "":
# #     print("No income statement data returned for TCS.NS.")
# # else:
# #     print("Income statement data found.")



# base_path = "./eval_results"

# # Step 1: find the company folder
# company_folder = os.listdir(base_path)[0]

# # Step 2: go into TradingAgentsStrategy_logs
# logs_path = os.path.join(base_path, company_folder, "TradingAgentsStrategy_logs")

# # Step 3: get the json file
# json_file = os.listdir(logs_path)[0]

# json_path = os.path.join(logs_path, json_file)

# print("Reading file:", json_path)

# # Step 4: open json
# with open(json_path, "r") as f:
#     data = json.load(f)

# # JSON structure contains date as key
# data = list(data.values())[0]

# company = data["company_of_interest"]
# analysis_date = data["trade_date"]

# # Step 5: separate agent reports
# agent_reports = {
#     "market": data.get("market_report"),
#     "sentiment": data.get("sentiment_report"),
#     "news": data.get("news_report"),
#     "fundamentals": data.get("fundamentals_report"),
#     "research_manager": data.get("investment_plan"),
#     "trader": data.get("trader_investment_decision"),
#     "risk_manager": data.get("final_trade_decision")
# }

# # Step 6: print reports
# for agent, report in agent_reports.items():
#     print("\n==================================================================================")
#     print("Agent:", agent)
#     print("====================================================================================")
#     print(report)

# print(agent_reports)

# query = """
#     INSERT INTO agent_reports( company_name, report_date, agent_name, report)
#     VALUES (%s,%s,%s,%s)
#     """

# for agent, report in agent_reports.items():
#     if report:
#         execute_query(query, (company, analysis_date, agent, report)) 


from dotenv import load_dotenv
load_dotenv()
import os
company_symbol = str(os.environ.get("COMPANY_NAME"))
print(company_symbol)
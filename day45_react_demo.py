import yaml
from dotenv import load_dotenv

from agent.chat_agent import ChatAgent
from tools.weather_tool import get_weather
from tools.calculator_tool import calculator
from tools.search_tool import search_web

load_dotenv()

with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


agent = ChatAgent(config)

agent.register_tool("get_weather", get_weather)
agent.register_tool("calculator", calculator)
agent.register_tool("search_web", search_web)

question = """
你必须通过调用工具来回答问题, 不能凭自己的知识直接回答。
问题: 北京今天多少度? 换算成华氏度是多少？
"""
print("提问: " + question)
answer = agent.react(question, max_steps=5)
print("最终回答: " + answer)
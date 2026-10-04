import os
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
agent.register_tool("calcultor", calculator)
agent.register_tool("search_web", search_web)

test_question = [
    "北京今天天气怎么样? ",
    "帮我算一下 123 * 456 = ? ",
    "帮我介绍一下量子计算量子计算最新研究进展",
]

for q in test_question:
    print("\n" + "=" * 60)
    print(f"用户提问: {q}")
    print("=" * 60)
    answer = agent.chat(q)
    print(f"Agent 回答: {answer}")

print("\nDay 44 多工具路由演示完成! ")
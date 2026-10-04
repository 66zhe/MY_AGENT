import yaml
from dotenv import load_dotenv

from agent.chat_agent import ChatAgent
from tools.weather_tool import get_weather
from tools.search_tool import search_web
from tools.calculator_tool import calculator
from tools.rag_tool import search_knowledge_base

load_dotenv()

with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

agent = ChatAgent(config)

agent.register_tool("get_weather", get_weather)
agent.register_tool("calculator", calculator)
agent.register_tool("search_web", search_web)
agent.register_tool("search_knowledge_base", search_knowledge_base)

test_question = [
    "我们上传的资料里, 文章提到了哪些缓解焦虑的方法? ",
    "北京今天多少度? ",
    "你好啊, 今天心情不错。",
]

for q in test_question:
    print("\n" + "=" * 50)
    print("提问：" + q)
    answer = agent.react(q, max_steps=5)
    print("最终回答：" + answer)
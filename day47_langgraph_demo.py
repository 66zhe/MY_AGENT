import os
import yaml
from typing import TypedDict
from zhipuai import ZhipuAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("未找到环境变量 ZHIPUAI_API_KEY")
client = ZhipuAI(api_key=api_key)

with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
model = config.get('llm', {}).get('name', 'glm-4-flash')
system_prompt = config.get('system_prompt', "你是一个有用的助手")

class GraphState(TypedDict):
    user_input: str
    answer: str

def call_llm(state: GraphState) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["user_input"]},
    ]
    resp = client.chat.completions.create(model=model,messages=messages)
    return {"answer": resp.choices[0].message.content}

def polish(state: GraphState) -> dict:
    return {"answer": state["answer"] + "\n\n(*^▽^*) 以上是我认真回答的哦~"}

builder = StateGraph(GraphState)
builder.add_node("call_llm", call_llm)
builder.add_node("polish", polish)
builder.add_edge(START, "call_llm")
builder.add_edge("call_llm", "polish")
builder.add_edge("polish", END)
app = builder.compile()

final_state = app.invoke({"user_input": "用一句话介绍一下你自己", "answer": ""})
print("最终回答: \n", final_state["answer"])
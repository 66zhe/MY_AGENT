from dotenv import load_dotenv
load_dotenv()

import os, time, yaml
import tools.weather_tool as wt
from typing import TypedDict
from zhipuai import ZhipuAI
from langgraph.graph import StateGraph, START, END

api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("未找到环境变量 ZHIPUAI_API_KEY")
client = ZhipuAI(api_key=api_key)

with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
model = config.get('llm', {}).get('name', "glm-4-flash")

def get_weather_offline(city: str) -> str:
    raise ConnectionError("模拟断网: 无法连接天气服务")

def call_tool_with_retry(func, retries=2, backoff=1, **kwargs):
    for attempt in range(retries + 1):
        try:
            return func(**kwargs), None
        except Exception as e:
            print(f"[重试] {func.__name__} 第{attempt+1}次失败: {e}")
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
    return None, f"{func.__name__} 重试{retries+1}次仍失败"

class FlowState(TypedDict):
    city: str
    tool_result: str
    reply: str
    error: str

def tool_node(state: FlowState) -> dict:
    func = wt.get_weather
    result, err = call_tool_with_retry(func, retries=2, city=state["city"])
    if err:
        return {"error": err}
    return {"tool_result": result}

def router(state: FlowState) -> str:
    """条件边的路由函数: 出错去error, 否则正常往下。"""
    return "error" if state.get("error") else "answer"

def answer_node(state: FlowState) -> dict:
    msgs = [
        {"role": "system", "content": config.get('system_prompt', "你是一个助手")},
        {"role": "user", "content": f"用户问: {state['city']}天气怎么样? 工具返回: {state['tool_result']}"},
    ]
    resp = client.chat.completions.create(model=model, messages=msgs)
    return {"reply": resp.choices[0].message.content}

def error_node(state: FlowState) -> dict:
    return {"reply": f" 【降级提示】 {state['error']}, 暂时无法提供天气信息, 这一步先跳过。"}

builder = StateGraph(FlowState)
builder.add_node("tool", tool_node)
builder.add_node("answer", answer_node)
builder.add_node("error", error_node)
builder.add_edge(START, "tool")
builder.add_conditional_edges("tool", router, {"answer": "answer", "error": "error"})
builder.add_edge("answer", END)
builder.add_edge("error", END)
app = builder.compile()

original_func = wt.get_weather
for label, use_bad in [("正常工具场景", False), ("断网模拟场景", True)]:
    print(f"\n===== 场景: {label} =====")

    if use_bad:
        wt.get_weather = get_weather_offline

    final = app.invoke({"city": "西安", "tool_result": "", "reply": "", "error": ""})

    if use_bad:
        wt.get_weather = original_func

    print("最终回复: ", final["reply"])
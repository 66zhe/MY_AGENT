from dotenv import load_dotenv
load_dotenv()

import os
import yaml
from typing import TypedDict
from zhipuai import ZhipuAI
from langgraph.graph import StateGraph, START, END
from agent.writer_agent import WriterAgent
from rag.rag_pipeline import rag_answer

api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("未找到环境变量 ZHIPUAI_API_KEY")
client = ZhipuAI(api_key=api_key)

with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
model = config.get('llm', {}).get('name', "glm-4-flash")

config['api_key'] = api_key
writer = WriterAgent(config)

class PipelineState(TypedDict):
    topic: str
    research_notes: str
    article: str

RESEARCHER_PROMPT = (
    "你是一名严谨的研究员。你会拿到从知识库检索到的原始资料。 "
    "请把资料提炼、去重、整理成条理清晰的要点, 作为写作素材。"
    "只依据给定资料, 不要凭空补充。"
)

def rearch_node(state: PipelineState) -> dict:
    raw = rag_answer(f"请介绍: {state['topic']}")
    messages = [
        {"role": "system", "content": RESEARCHER_PROMPT},
        {"role": "user", "content": f"主题: {state['topic']}\n检索到的原始资料: \n{raw}\n请整理为写作要点。"},
    ]
    resp = client.chat.completions.create(model=model, messages=messages)
    return {"research_notes": resp.choices[0].message.content}

def write_node(state: PipelineState) -> dict:
    article = writer.write_article(state["topic"], state["research_notes"])
    return {"article": article}

builder = StateGraph(PipelineState)
builder.add_node("research", rearch_node)
builder.add_node("write", write_node)
builder.add_edge(START, "research")
builder.add_edge("research", "write")
builder.add_edge("write", END)
app = builder.compile()

final_state = app.invoke({"topic": "RAG技术", "research_notes": "", "article": ""})
print("=== 研究院资料 ===\n", final_state["research_notes"][:300], "...\n")
print("=== 最终文章 ===\n", final_state["article"])
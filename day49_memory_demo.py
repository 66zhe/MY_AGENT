from dotenv import load_dotenv
load_dotenv()

import os, operator, yaml
from typing import TypedDict, Annotated
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
    chat_history: Annotated[list, operator.add]

RESEARCHER_PROMPT = (
    "你是研究员。你会拿到从知识库检索到的原始资料, 以及此前的对话历史。"
    "请结合历史与资料, 把要点整理成条理清晰的写作素材; 只依据给定资料, 不要凭空补充。"
)

def research_node(state: PipelineState) -> dict:
    raw = rag_answer(f"请介绍: {state['topic']}")
    msgs = [{"role": "system", "content": RESEARCHER_PROMPT}]
    msgs += state["chat_history"][: -1] or []
    msgs.append({"role": "user",
        "content": f"主题: {state['topic']}\n检索到的原始资料: \n{raw})\n请整理为写作要点。"})
    resp = client.chat.completions.create(model=model, messages=msgs)
    notes = resp.choices[0].message.content
    return {"research_notes": notes,
        "chat_history": [{"role": "user", "content": f"[研究院资料] {notes[:120]}..."}]}

def write_node(state: PipelineState) -> dict:
    article = writer.write_article(state["topic"], state["research_notes"])
    return {"article": article,
        "chat_history": [{"role": "assistant", "content": f"[成文] {article[:120]}..."}]}

builder = StateGraph(PipelineState)
builder.add_node("research", research_node)
builder.add_node("write", write_node)
builder.add_edge(START, "research")
builder.add_edge("research", "write")
builder.add_edge("write", END)
app = builder.compile()

history = []
for topic in ["RAG技术", "把刚才那篇文章里'应用场景'再展开讲讲"]:
    final = app.invoke({"topic": topic, "research_notes": "", "article": "", "chat_history": history})
    history = final["chat_history"]
    print(f"=== 主题: {topic} ===")
    print("研究员资料: \n", final["research_notes"][:200], "...\n")
    print("最终文章: \n", final["article"][:300], "...\n")
    print(f"[历史累积 {len(history)} 条]\n")
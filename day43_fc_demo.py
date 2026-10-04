"""
Day 43 实验C：Function Calling 多工具路由演示
目的：注册两个工具，测试模型能否根据用户问题，智能判断并选择正确的工具。
"""
import os
import json
from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()

# ========== 1. 初始化客户端 ==========
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("未找到环境变量 ZHIPUAI_API_KEY")

client = ZhipuAI(api_key=api_key)
MODEL = "glm-4-flash"


# ========== 2. 定义两个工具函数 ==========
def get_weather(city: str) -> str:
    """查询指定城市的实时天气"""
    weather_data = {
        "北京": "晴天,28℃,微风",
        "上海": "多云,26℃,东南风3级",
        "西安": "阴,24℃,无风"
    }
    return weather_data.get(city, f"暂无{city}的天气数据")


def search_web(query: str) -> str:
    """模拟全网搜索工具，搜索关于某个主题的详细资料"""
    mock_database = {
        "量子计算": "量子计算是一种遵循量子力学规律调控量子信息单元进行计算的新型计算模式。",
        "人工智能": "计算机科学的一个分支，致力于创造能模仿人类智能行为的系统。"
    }
    for key in mock_database:
        if key in query:
            return mock_database[key]
    return f"未找到关于'{query}'的详细资料"


# ========== 3. 工具描述：把两个工具一起注册进去 ==========
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如北京、上海、西安"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "模拟全网搜索工具，搜索关于某个主题的详细资料",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# ========== 4. 构造对话消息（测试实验C：量子计算） ==========
messages = [
    {"role": "user", "content": "用 search_web 搜索一下量子计算的发展现状"}
]

# ========== 5. 第一次调用 API ==========
print("=" * 60)
print("【第一次调用】模型面临选择：调用天气 还是 搜索工具？")
print("=" * 60)

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=tools
)

message = response.choices[0].message


# ========== 6. 判断模型是否要求调用工具 ==========
if hasattr(message, 'tool_calls') and message.tool_calls:
    print("\n模型说：'我需要调用工具！'")
    print(f"模型返回的 tool_calls 数量: {len(message.tool_calls)}")

    for tc in message.tool_calls:
        print(f"\n  函数名: {tc.function.name}")
        print(f"  参数(JSON): {tc.function.arguments}")
        print(f"  调用ID: {tc.id}")

    # ========== 7. 手动执行工具（动态路由） ==========
    first_call = message.tool_calls[0]
    func_name = first_call.function.name
    func_args = json.loads(first_call.function.arguments)

    print(f"\n{'=' * 60}")
    print(f"【执行工具】路由到 -> {func_name}({func_args})")
    print("=" * 60)

    # 动态分发：模型选了谁，就执行谁
    if func_name == "get_weather":
        tool_result = get_weather(**func_args)
    elif func_name == "search_web":
        tool_result = search_web(**func_args)
    else:
        tool_result = "未知工具"

    print(f"工具返回结果: {tool_result}")

    # ========== 8. 把模型意图 + 工具结果塞进 messages ==========
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": first_call.id,
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": first_call.function.arguments
                }
            }
        ]
    })

    messages.append({
        "role": "tool",
        "tool_call_id": first_call.id,
        "content": str(tool_result)
    })

    # ========== 9. 第二次调用 API：生成最终回答 ==========
    print(f"\n{'=' * 60}")
    print("【第二次调用】把搜索到的量子计算资料发回给模型")
    print("=" * 60)

    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools
    )

    final_answer = final_response.choices[0].message.content
    print(f"\n模型最终回答: {final_answer}")

else:
    print(f"\n模型直接回答（未调用工具）: {message.content}")

print("\n" + "=" * 60)
print("Day 43 实验C 演示完成！")
print("=" * 60)
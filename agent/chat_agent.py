import os
import json
import logging
import inspect
from typing import List, Dict, Any, Callable

from zhipuai import ZhipuAI
from utils.logger import get_logger

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("ChatAgent")

class ChatAgent:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 Agent
        :param config: 从 config.yaml 加载的配置字典
        """
        # 1. 初始化客户端
        api_key = os.getenv("ZHIPUAI_API_KEY")
        if not api_key:
            raise ValueError("未找到环境变量 ZHIPUAI_API_KEY")
        
        self.client = ZhipuAI(api_key=api_key)
        self.model = config.get('model', {}).get('name', "glm-4-flash")
        
        # 2. 初始化对话历史，并注入 System Prompt (人设)
        system_prompt_content = config.get('system_prompt', "你是一个有用的助手")
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt_content}
        ]
        
        # 3. 初始化工具容器
        self.tools_map: Dict[str, Callable] = {}
        self.tools_definition: List[Dict[str, Any]] = []
        
        logger.info(f"Agent 初始化完成，模型：{self.model}")
        logger.debug(f"当前人设：{system_prompt_content[:50]}...")

    def register_tool(self, name: str, func: Callable):
        """
        注册工具：保存函数引用，并自动生成符合 API 格式的描述
        """
        # 1. 保存函数对象
        self.tools_map[name] = func
        
        # 2. 自动生成工具描述 (Tools Definition)
        sig = inspect.signature(func)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            # 简单推断类型
            param_type = "string"
            if param.annotation == int: param_type = "integer"
            elif param.annotation == float: param_type = "number"
            elif param.annotation == bool: param_type = "boolean"
            
            properties[param_name] = {
                "type": param_type,
                "description": f"参数 {param_name}"
            }
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        tool_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": func.__doc__ or f"调用 {name} 函数",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
        
        self.tools_definition.append(tool_def)
        logger.info(f"已注册工具：{name}")

    def chat(self, user_input: str) -> str:
        """
        核心交互方法
        """
        # 1. 添加用户消息
        self.messages.append({"role": "user", "content": user_input})
        logger.info(f"收到用户输入：{user_input}")

        try:
            # 2. 第一次调用：让模型决定是否需要工具
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools_definition if self.tools_definition else None
            )
            
            message = response.choices[0].message
            
            # 3. 判断模型意图
            if message.tool_calls:
                # --- 情况一：模型想要调用工具 ---
                logger.info(f"模型请求调用工具：{message.tool_calls}")
                
                # 将 Assistant 的意图存入历史
                self.messages.append(message.model_dump(exclude_none=True))
                
                # 执行工具
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"正在执行本地函数：{func_name}，参数：{func_args}")
                    
                    if func_name in self.tools_map:
                        func_result = self.tools_map[func_name](**func_args)
                    else:
                        func_result = f"错误：找不到函数 {func_name}"
                    
                    # 将工具执行结果存入历史
                    self.messages.append({
                        "role": "tool",
                        "content": str(func_result),
                        "tool_call_id": tool_call.id
                    })
                
                # 4. 第二次调用：带着工具结果，让模型生成最终回复
                second_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.tools_definition if self.tools_definition else None
                )
                final_reply = second_response.choices[0].message.content
                
            else:
                # --- 情况二：模型直接回复 ---
                final_reply = message.content
            
            # 5. 将最终回复存入历史
            self.messages.append({"role": "assistant", "content": final_reply})
            logger.info(f"Agent 回复：\n{final_reply}")
            return final_reply

        except Exception as e:
            logger.error(f"发生错误：{e}")
            return f"出错了：{str(e)}"
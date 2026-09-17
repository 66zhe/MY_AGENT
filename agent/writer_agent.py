import logging
from typing import Dict, Any
from zhipuai import ZhipuAI
from utils.logger import get_logger

logger = get_logger("WriterAgent")

class WriterAgent:
    """
    负责将研究员提供的原始资料，撰写成结构清晰的文章。
    """
    def __init__(self, config: Dict[str, Any]):
        api_key = config.get('api_key')
        if not api_key:
             raise ValueError("WriterAgent 未获取到 API Key")
        
        self.client = ZhipuAI(api_key=api_key)
        self.model = config.get('model', {}).get('name', "glm-4-flash")
        self.system_prompt = config.get('writer_prompt', "你是一个专业的科技专栏作家。")
        
        logger.info(f"WriterAgent 初始化完成，模型：{self.model}")

    def write_article(self, topic: str, research_data: str) -> str:
        """
        核心方法：接收主题和研究资料，生成文章
        """
        logger.info(f"收到写作任务：主题《{topic}》")
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"请根据以下关于【{topic}】的研究资料，写一篇总结文章：\n\n{research_data}"}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            article = response.choices[0].message.content
            logger.info("文章撰写完成")
            return article
        except Exception as e:
            logger.error(f"写作过程中发生错误: {e}")
            return f"写作失败：{str(e)}"
import os
import yaml
from dotenv import load_dotenv

# 导入 Agent 类
from agent.chat_agent import ChatAgent      
from agent.writer_agent import WriterAgent  

# 导入工具
from tools.weather_tool import get_weather
from tools.search_tool import search_web  # <--- 新增：导入搜索工具

from utils.logger import get_logger

# 加载环境变量
load_dotenv()

logger = get_logger("MainOrchestrator")

def load_config(path="config/config.yaml") -> dict:
    """加载 YAML 配置文件"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_collaboration_pipeline():
    """
    双 Agent 协作主流程
    """
    logger.info("=== 双 Agent 协作系统启动 ===")
    
    # 1. 加载配置
    config = load_config()
    config['api_key'] = os.getenv("ZHIPUAI_API_KEY")

    # 2. 初始化两个 Agent
    # 研究员：负责查资料
    researcher = ChatAgent(config)
    # 给研究员配备“千里眼”和“顺风耳”
    researcher.register_tool('get_weather', get_weather) 
    researcher.register_tool('search_web', search_web)   # <--- 新增：注册搜索工具
    
    # 写手：负责写文章
    writer = WriterAgent(config)

    print("\n" + "="*50)
    print("🤖 多智能体协作系统已就绪 (Researcher + Writer)")
    print("💡 提示：你可以问我“介绍一下量子计算”或者“西安天气怎么样”")
    print("输入 'quit' 退出")
    print("="*50)

    # 3. 主循环
    while True:
        try:
            user_input = input("\n👤 用户: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            if not user_input.strip():
                continue

            topic = user_input.strip()
            logger.info(f"--- 开始处理新任务：{topic} ---")

            # --- 阶段一：研究员工作 ---
            print(f"\n[系统] 🕵️‍♂️ ResearcherAgent 正在全网搜集关于《{topic}》的资料...")
            
            # 这里的 Prompt 很关键，它指挥 ChatAgent 去调用工具
            research_instruction = (
                f"用户想了解关于【{topic}】的信息。"
                "请务必调用你拥有的工具（search_web 或 get_weather）来获取最准确的实时数据。"
                "获取到数据后，请直接返回工具的原始结果，不要自己编造，也不要直接回答用户。"
            )
            
            raw_research_data = researcher.chat(research_instruction)
            
            # --- 阶段二：写手工作 ---
            print(f"\n[系统] ✍️ WriterAgent 正在整理资料并撰写文章...")
            
            final_article = writer.write_article(topic, raw_research_data)
            
            # --- 阶段三：输出结果 ---
            print("\n" + "="*20 + " 📄 最终报告 " + "="*20)
            print(final_article)
            print("="*50)
            logger.info(f"任务 {topic} 完成。")

        except Exception as e:
            logger.error(f"主程序运行出错: {e}")
            print(f"系统错误: {e}")

if __name__ == '__main__':
    run_collaboration_pipeline()
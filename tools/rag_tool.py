from rag.rag_pipeline import rag_answer

def search_knowledge_base(query:str) -> str:
    """
    在 【用户自己上传/导入的本地知识库文档】中检索并回答问题。
    当用户询问的是"我们之前上传的资料/文档里写了什么/知识库里的内容"
    这类必须查本地文档才能回答的问题时, 必须调用本工具, 不要凭记忆回答。
    注意: 天气、数学计算类问题不要用本工具 (那些有专门的工具)。
    query 为要查询的问题或关键词, 如"文章里提到的缓解焦虑的方法"。
    """

    try:
        answer = rag_answer(query)
        return str(answer)
    except Exception as e:
        return f"知识库检索失败: {e}"
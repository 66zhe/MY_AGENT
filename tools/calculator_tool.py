def calculator(expression: str) -> str:
    """
    进行数学计算, 输入一个数学表达式字符串, 返回计算结果。
    支持加减乘除与括号, 例如 (123 + 456) * 2、3.14 * 10、28*9/5+32。
    当用户或前面的工具给出了数字, 需要你进行加减乘除计算时调用本工具。
    """
    try:
        allowed = set("0123456789+-*/(). %")
        if not all(c in allowed for c in expression):
            return f"表达式包含不支持的字符, 仅支持数字与 + - * / () %: "
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败: {e}" 
def get_weather(city:str) -> str:
    """
    查询指定城市的实时天气
    """

    weather_data = {
        "北京":"晴天,28℃,微风",
        "上海":"多云,26℃,东南风3级",
        "西安":"阴,24℃,无风"
    }
    return weather_data.get(city,f"暂无{city}的天气数据")
import openai
import requests

openai.api_key = "YOUR_OPENAI_API_KEY"

def get_current_weather(location: str, unit: str = "metric"):
    """
    调用天气 API 获取当前天气
    :param location: 城市名称
    :param unit: 温度单位（metric 表示摄氏度，imperial 表示华氏度）
    :return: 字典格式的天气数据，包含温度(temperature)
    """
    api_key = "YOUR_OPENWEATHERMAP_API_KEY"
    
    # 构建API请求URL（使用f-string格式化参数）
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&units={unit}&appid={api_key}"
    
    # 发送HTTP GET请求并解析JSON响应
    response = requests.get(url)
    data = response.json()
    
    # 返回天气信息
    return {
        "temperature": data["main"]["temp"],
        "description": data["weather"][0]["description"]
    }

# 定义Function Call的参数，注册函数给大模型，需符合模型要求的格式，如JSON Schema
functions = [
    {
        "name": "get_current_weather",
        "description": "获取指定城市的当前天气",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名称"
                },
                "unit": {
                    "type": "string",
                    "description": "温度单位（metric 或 imperial）",
                    "default": "metric"
                }
            },
            "required": ["location"]
        }
    }
]

# 用户输入
user_input = "北京今天的天气如何？"

# 调用 OpenAI API，启用 Function Call
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": user_input}
    ],
    functions=functions,
    # 自动决定是否调用函数
    function_call="auto"  
)

# 检查是否需要调用函数
if response.choices[0].finish_reason == "function_call":
    function_args = response.choices[0].message.function_call.arguments
    function_name = response.choices[0].message.function_call.name

     # 调用外部函数
    if function_name == "get_current_weather":
        weather_result = get_current_weather(
            location=function_args["location"],
            unit=function_args.get("unit", "metric")
        )

        # 将结果返回给模型
        second_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response.choices[0].message.content},
                {"role": "function", "name": function_name, "arguments": function_args},
                {"role": "function", "name": function_name, "content": weather_result}
            ]
        )
        print(second_response.choices[0].message.content)
else:
    print(response.choices[0].message.content)
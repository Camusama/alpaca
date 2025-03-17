import json 
from tqdm import tqdm
import requests
from typing import List
from openai import OpenAI

##### API 配置 #####


client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model = 'qwen-max'
print(model)

def generate_response(ins):
    stream = False
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": ins,
                        }
                    ],
                }
            ],
            model=model,
            max_completion_tokens=4096,
            stream=stream,
        )
        res = ''
        if stream:
            for chunk in chat_completion:
                res += chunk.choices[0].delta.content
        else:
            res = chat_completion.choices[0].message.content
        return res
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

input_file_path = "input.json" # 输入文件名
output_file_path = "output.json" # 输出文件名

# 读取已存在的输出文件
existing_data = []
existing_instructions = set()
try:
    with open(output_file_path) as f:
        existing_data = json.load(f)
        existing_instructions = {item["instruction"] for item in existing_data}
except FileNotFoundError:
    pass

with open(input_file_path) as fp:
    data = json.load(fp)

# 过滤出未处理的数据
unprocessed_data = [d for d in data if d["instruction"] not in existing_instructions]

pbar = tqdm(total=len(unprocessed_data))
new_data = existing_data.copy()  # 使用已存在的数据作为基础
retry_items = []

for d in unprocessed_data:
    prompt = d["instruction"]
    output = generate_response(prompt)
    if output:
        temp = {
            "instruction": prompt,
            "output": output
        }
        new_data.append(temp)
    else:
        retry_items.append(d)
    pbar.update(1)
pbar.close()

# 自动重试机制
retry_count = 1
while retry_items:
    print(f"\n第 {retry_count} 轮重试，剩余 {len(retry_items)} 个项目...")
    current_retry = retry_items.copy()
    retry_items = []
    
    pbar = tqdm(total=len(current_retry))
    for d in current_retry:
        prompt = d["instruction"]
        output = generate_response(prompt)
        if output:
            temp = {
                "instruction": prompt,
                "output": output
            }
            new_data.append(temp)
        else:
            retry_items.append(d)
        pbar.update(1)
    pbar.close()
    retry_count += 1

output_file_path = "output.json" # 输出文件名
with open(output_file_path, 'w') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
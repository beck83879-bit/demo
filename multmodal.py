import base64
import json
import re
import requests
import math
from io import BytesIO

from openai import OpenAI
from PIL import Image, ImageDraw

prompt_ins = '''
You are a Multimodal Question Answering Agent for complex tasks. You have access to the following tools:
1. <search_visual>query</search_visual>: Retrieve relevant images or visual diagrams based on the query.
2. <search_text>query</search_text>: Retrieve relevant text passages, facts, or definitions based on the query.
3. <search_table>query</search_table>: Retrieve structured tabular data if needed.
4. <crop>[x1, y1, x2, y2]</crop>: Zoom into, crop or focus on the region of an image with coordinates for clearer view.
5. <text_rewrite>text</text_rewrite>: Rephrase the given text if needed.
6. <answer>your final answer here
'''

class VRAG:
    def __init__(self, 
                base_url='http://localhost:8000/v1', 
                search_url='https://api.bing.microsoft.com/v7.0/images/search',
                generator=True,
                api_key='EMPTY'):
        
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = '/data/ykb/qwen3-vl-2b-instruct'
        self.search_url = search_url

        self.max_pixels = 512 * 28 * 28
        self.min_pixels = 256 * 28 * 28
        self.repeated_nums = 1
        self.max_steps = 10

        self.generator = generator

    def process_image(self, image):
        if isinstance(image, dict):
            image = Image.open(BytesIO(image['bytes']))
        elif isinstance(image, str):
            image = Image.open(image)

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        byte_stream = BytesIO()
        image.save(byte_stream, format="JPEG")
        byte_array = byte_stream.getvalue()
        base64_encoded_image = base64.b64encode(byte_array)
        base64_string = base64_encoded_image.decode("utf-8")
        base64_qwen = f"data:image;base64,{base64_string}"

        return image, base64_qwen
    
    def search(self, query):
        import os
        api_key = os.getenv('SERPER_API_KEY')
    
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
    
        search_query = query[0] if isinstance(query, list) else query
        payload = json.dumps({"q": search_query, "num": 5})
    
        try:
            response = requests.post(
                'https://google.serper.dev/images',
                headers=headers,
                data=payload,
                timeout=10
            )
            results = response.json()
            return [img['imageUrl'] for img in results.get('images', [])[:5]]
        except Exception as e:
            print(f"搜索失败: {e}")
            return []
    def search_text(self, query):
        """检索文本内容"""
        import os
        api_key = os.getenv('SERPER_API_KEY')
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        search_query = query[0] if isinstance(query, list) else query
        payload = json.dumps({"q": search_query, "num": 5})
        try:
            response = requests.post(
                'https://google.serper.dev/search',  # 改为通用搜索而非图像搜索
                headers=headers,
                data=payload,
                timeout=10
            )
            results = response.json()
            return [result['snippet'] for result in results.get('organic', [])[:5]]
        except Exception as e:
            print(f"文本搜索失败: {e}")
            return []
    def search_table(self, query):
        """检索表格内容"""
        pass
    def run(self, question):
        self.image_raw = []
        self.image_input = []
        self.image_path = []
        prompt = prompt_ins.format(question=question)
        messages = [dict(
            role="user",
            content=[
                {
                    "type": "text",
                    "text": prompt,
                }
            ]
        )]

        max_steps = self.max_steps
        while True:
            ## assistant
            response = self.client.chat.completions.create(
                model="/data/ykb/qwen3-vl-2b-instruct",
                messages=messages,
                stream=False,
                max_tokens=2048,
                extra_body={
                    "chat_template_kwargs": {
                    "enable_thinking": False  # 禁用thinking模式
        }}
            )
            response_content = response.choices[0].message.content
            #增加调试输出
            print(f"\n【调试模型输出:{response_content[:200]}\n")
            messages.append(dict(
                role="assistant",
                content=[{
                    "type": "text",
                    "text": response_content
                }]
            ))
            ## think
            pattern = r'<think>(.*?)</think>'
            match = re.search(pattern, response_content, re.DOTALL)
            ## think
            # 新增：判断匹配是否成功，避免 NoneType 错误
            if match:
                thought = match.group(1).strip()
                full_match = match.group(0)
            else:
    # 如果没有think标签，提取answer/search/bbox标签之前的内容
                action_pattern = r'<(answer|search|bbox)>'
                action_match = re.search(action_pattern, response_content)
                if action_match:
                    thought = response_content[:action_match.start()].strip()
                    full_match = thought
                else:
                    thought = response_content.strip()  # 全部内容作为thought
                    full_match = thought

            if self.generator:
                yield 'think', thought, full_match  # 这里改用上面定义的 full_match

            ## opration
            pattern = r'<(search|answer|bbox)>(.*?)</\1>'
            match = re.search(pattern, response_content, re.DOTALL)
            if match:
                raw_content = match.group(0)
                content = match.group(2).strip()
                action = match.group(1)
            else:
                content = ''
                action = None

            ## whether end
            if action == 'answer':
                if self.generator:
                    yield 'answer', content, raw_content
                return  # 结束循环
    
            if max_steps == 0:
                if self.generator:
                    yield 'answer', 'Sorry, I can not retrieval something about the question.', ''
                return

            # 其他action继续处理
            if self.generator and action:
                yield action, content, raw_content


            ## action
            if action == 'search':
                search_results = self.search(content)
                while len(search_results) > 0:
                    image_path = search_results.pop(0)
                    if self.image_path.count(image_path) >= self.repeated_nums:
                        continue
                    else:
                        self.image_path.append(image_path)
                        break
                
                image_raw = Image.open(image_path)
                image_input, img_base64 = self.process_image(image_raw)
                user_content=[{
                    'type': 'image_url',
                    'image_url': {
                        'url': img_base64
                    }
                }]
                self.image_raw.append(image_raw)
                self.image_input.append(image_input)
                if self.generator:
                    yield 'search_image', self.image_input[-1], raw_content
            elif action == 'bbox':
                bbox = json.loads(content)
                input_w, input_h = self.image_input[-1].size
                raw_w, raw_h = self.image_raw[-1].size
                crop_region_bbox = bbox[0] * raw_w / input_w, bbox[1] * raw_h / input_h, bbox[2] * raw_w / input_w, bbox[3] * raw_h / input_h
                pad_size = 56
                crop_region_bbox = [max(crop_region_bbox[0]-pad_size,0), max(crop_region_bbox[1]-pad_size,0), min(crop_region_bbox[2]+pad_size,raw_w), min(crop_region_bbox[3]+pad_size,raw_h)]
                crop_region = self.image_raw[-1].crop(crop_region_bbox)
                image_input, img_base64 = self.process_image(crop_region)
                user_content=[{
                    'type': 'image_url',
                    'image_url': {
                        'url': img_base64
                    }
                }]
                self.image_raw.append(crop_region)
                self.image_input.append(image_input)

                if self.generator:
                    image_to_draw = self.image_input[-2].copy()
                    draw = ImageDraw.Draw(image_to_draw)
                    draw.rectangle(bbox, outline=(160, 32, 240), width=7)
                    yield 'crop_image', self.image_input[-1], image_to_draw

            max_steps -= 1
            if max_steps == 0:
                user_content.append({
                    'type': 'text',
                    'text': 'please answer the question now with answer in <answer> ... </answer>' 
                })
            messages.append(dict(
                role='user',
                content=user_content
            ))

if __name__ == '__main__':
    agent = VRAG()
    generator = agent.run('How are u?')
    while True:
        print(next(generator))
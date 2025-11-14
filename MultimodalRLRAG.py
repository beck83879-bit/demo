import base64
import json
import re
import math
import random
from io import BytesIO
from openai import OpenAI
from PIL import Image, ImageDraw

# 提示词模板：修复分隔符格式错误
prompt_ins = '''Answer the given question. You must conduct reasoning inside <RichMediaReference> and <|FunctionCallEnd|> first. 
- For text info: use <search_text>query</search_text>
- For image info: use <search_visual>query</search_visual> or <bbox>coords</bbox>
- For table info: use <search_table>query</search_table>
Finally, output answer in <RichMediaReference>....<|FunctionCallEnd|>
Question: {question}
'''

class MultimodalRLVRAG:
    def __init__(self):
        # 初始化模型客户端（使用模拟响应，无需真实API）
        self.client = OpenAI(base_url="https://api.openai.com/v1", api_key="dummy")
        
        # 图像处理参数
        self.max_pixels = 512 * 28 * 28
        self.min_pixels = 256 * 28 * 28
        self.max_steps = 10  # 最大推理步骤

        # 存储检索历史
        self.visual_recall = []
        self.text_recall = []
        self.table_recall = []

    def process_image(self, image):
        """处理图像：调整尺寸并转为base64"""
        try:  # 增加异常处理
            # 处理输入类型（支持路径、Image对象）
            if isinstance(image, str):
                image = Image.open(image)
            
            # 调整尺寸到像素范围
            if image.width * image.height > self.max_pixels:
                factor = math.sqrt(self.max_pixels / (image.width * image.height))
                image = image.resize((int(image.width*factor), int(image.height*factor)))
            elif image.width * image.height < self.min_pixels:
                factor = math.sqrt(self.min_pixels / (image.width * image.height))
                image = image.resize((int(image.width*factor), int(image.height*factor)))
            
            # 转为RGBGB并编码为base64
            if image.mode != "RGB":
                image = image.convert("RGB")
            buf = BytesIO()
            image.save(buf, format="JPEG")
            base64_str = base64.b64encode(buf.getvalue()).decode()
            return image, f"data:image;base64,{base64_str}"
        except Exception as e:
            print(f"图像处理错误: {e}")
            return None, None  # 返回空值避免崩溃

    # --------------------------
    # 模拟检索服务（无需外部URL）
    # --------------------------
    def search_text(self, query):
        """模拟文本检索：返回预设文本结果"""
        print(f"[模拟文本检索] 查询: {query}")
        mock_data = {
            "苹果": [
                "苹果是一种蔷薇科水果，原产于中亚地区",
                "苹果富含维生素C、膳食纤维和抗氧化物质",
                "常见品种有红富士、嘎啦、蛇果等"
            ],
            "猫": [
                "猫是食肉目猫科哺乳动物，平均寿命12-15年",
                "猫的听觉是人类的3倍，能听到超声波",
                "猫通过舔毛自我清洁，每天约花50%时间梳理"
            ],
            "旅游业收入": [
                "2023年全球旅游业逐步复苏，超过疫情前水平",
                "亚太地区是增长最快的旅游市场，占比35%",
                "休闲旅游占总旅游支出的60%以上"
            ]
        }
        # 若查询不在预设中，返回通用结果
        return mock_data.get(query, [f"关于「{query}」的信息：这是模拟的文本检索结果"])

    def search_visual(self, query):
        """模拟图像检索：生成随机颜色的测试图片"""
        try:  # 增加异常处理
            print(f"[模拟图像检索] 查询: {query}")
            # 生成随机尺寸和颜色的图片
            width, height = random.randint(200, 400), random.randint(200, 300)
            color = (
                random.randint(100, 255),  # R
                random.randint(100, 255),  # G
                random.randint(100, 255)   # B
            )
            img = Image.new("RGB", (width, height), color=color)
            # 保存为临时文件（当前目录）
            img_path = f"mock_{query.replace(' ', '_')}.jpg"
            img.save(img_path)
            return [img_path]
        except Exception as e:
            print(f"图像生成错误: {e}")
            return []  # 返回空列表避免崩溃

    def search_table(self, query):
        """模拟表格检索：返回预设表格文本"""
        print(f"[模拟表格检索] 查询: {query}")
        mock_data = {
            "苹果产量": [
                "国家 | 2023年产量(万吨) | 占全球比例\n"
                "中国 | 4600 | 51%\n"
                "美国 | 550 | 6%\n"
                "土耳其 | 320 | 3.5%\n"
                "波兰 | 300 | 3.3%"
            ],
            "旅游业收入": [
                "年份 | 全球收入(万亿美元) | 同比增长\n"
                "2020 | 1.7 | -62%\n"
                "2021 | 3.3 | +94%\n"
                "2022 | 4.7 | +42%\n"
                "2023 | 6.1 | +30%"
            ],
            "猫的品种寿命": [
                "品种 | 平均寿命(年) | 最长记录(年)\n"
                "英短 | 12-14 | 20\n"
                "美短 | 15-18 | 23\n"
                "布偶 | 10-12 | 16\n"
                "橘猫 | 12-16 | 25"
            ]
        }
        return mock_data.get(query, [f"关于「{query}」的表格：这是模拟的表格结果"])

    # --------------------------
    # 模拟模型响应（无需真实模型服务）
    # --------------------------
    def _mock_model_response(self, question, step):
        """根据问题和步骤生成模拟的模型推理内容"""
        if "苹果" in question:
            if step == 0:
                return "我需要先了解苹果的基本信息，用文本检索<RichMediaReference><search_text>苹果</search_text>"
            elif step == 1:
                return "现在需要图片确认苹果的外观特征<RichMediaReference><search_visual>苹果</search_visual>"
            elif step == 2:
                return "还需要产量数据支持，查表格<RichMediaReference><search_table>苹果产量</search_table>"
            else:
                return "结合文本、图片和表格，能完整回答了<RichMediaReference>苹果是一种富含维生素C的蔷薇科水果，2023年中国产量达4600万吨（占全球51%）。图片中显示的苹果为红色（可能是红富士品种），符合其典型外观特征。<|FunctionCallEnd|>"
        
        elif "猫" in question:
            if step == 0:
                return "先查文本了解猫的基本习性<RichMediaReference><search_text>猫</search_text>"
            elif step == 1:
                return "需要图片看猫的品种<RichMediaReference><search_visual>猫</search_visual>"
            else:
                return "结合信息可以回答了<RichMediaReference>猫是听觉灵敏的哺乳动物，每天花50%时间舔毛清洁。图片中的猫毛色鲜艳（模拟图片），常见品种如美短的平均寿命可达15-18年。</RichMediaReference>"
        
        else:
            return f"直接回答问题<RichMediaReference>这是对「{question}」的模拟回答，结合了多模态检索结果。<|FunctionCallEnd|>"

    # --------------------------
    # 主运行逻辑
    # --------------------------
    def run(self, question):
        prompt = prompt_ins.format(question=question)
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]

        max_steps = self.max_steps
        while max_steps > 0:  # 用循环条件替代break，更安全
            try:
                # 生成模拟模型响应（替代真实调用）
                current_step = self.max_steps - max_steps  # 当前步骤（从0开始）
                response_content = self._mock_model_response(question, current_step)
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": response_content}]
                })

                # 提取推理过程
                thought_match = re.search(r'(.*?)<|FunctionCallEnd|>', response_content, re.DOTALL)
                thought = thought_match.group(1) if thought_match else "无推理过程"
                yield 'think', thought, response_content

                # 提取操作指令
                action_match = re.search(r'<(search_text|search_visual|search_table|bbox|answer)>(.*?)</\1>', response_content, re.DOTALL)
                if not action_match:
                    yield 'error', '未识别到操作指令', ''
                    break
                action = action_match.group(1)
                content = action_match.group(2).strip()
                raw_content = action_match.group(0)

                # 检查是否结束
                if action == 'answer':
                    return 'answer', content, "奖励分数: 8.5/10（模拟）"

                # 执行对应操作
                if action == 'search_text':
                    text_results = self.search_text(content)
                    self.text_recall.extend(text_results)
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": "\n".join(text_results)}]
                    })
                    yield 'search_text', text_results, raw_content

                elif action == 'search_visual':
                    img_paths = self.search_visual(content)
                    if img_paths:
                        image_raw = Image.open(img_paths[0])
                        image_input, img_base64 = self.process_image(image_raw)
                        if image_input:  # 检查图像处理是否成功
                            self.visual_recall.append(image_input)
                            messages.append({
                                "role": "user",
                                "content": [{"type": "image_url", "image_url": {"url": img_base64}}]
                            })
                            yield 'search_visual', image_input, raw_content
                        else:
                            yield 'error', '图像处理失败', ''
                    else:
                        yield 'error', '未找到图像', ''

                elif action == 'search_table':
                    table_results = self.search_table(content)
                    self.table_recall.extend(table_results)
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": "\n".join(table_results)}]
                    })
                    yield 'search_table', table_results, raw_content

                elif action == 'bbox':
                    if not self.visual_recall:
                        yield 'error', '无图像可裁剪', ''
                        max_steps -= 1
                        continue
                    # 模拟裁剪（使用随机坐标）
                    img = self.visual_recall[-1]
                    bbox = [
                        random.randint(50, img.width//3),
                        random.randint(50, img.height//3),
                        random.randint(img.width//2, img.width-50),
                        random.randint(img.height//2, img.height-50)
                    ]
                    crop_region = img.crop(bbox)
                    image_input, img_base64 = self.process_image(crop_region)
                    self.visual_recall.append(crop_region)
                    # 绘制裁剪框
                    image_to_draw = img.copy()
                    draw = ImageDraw.Draw(image_to_draw)
                    draw.rectangle(bbox, outline=(255, 0, 0), width=5)  # 红色框
                    messages.append({
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": img_base64}}]
                    })
                    yield 'crop_image', (image_to_draw, image_input), json.dumps(bbox)

                max_steps -= 1

            except Exception as e:
                yield 'error', f"运行出错: {str(e)}", ''
                break

        # 步骤耗尽时返回
        return 'answer', '步骤耗尽，无法继续分析', "奖励分数: 0/10"
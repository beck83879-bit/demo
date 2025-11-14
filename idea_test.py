import base64
import json
import re
import math
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
import requests
from openai import OpenAI
from unittest.mock import patch

#提示词模版：定义多模态推理和工具调用格式
prompt_ins='''Answer the given question step by step.
1.First,resaon inside and </RichMediaReference>.
2.If you need text info:use<search_text>query</search_text>
3.If you need image info:use<search>query</search>or<bbox>[x1,y1,x2,y2]</bbox>
4.Finally,output answer in <RichMediaResponse>...</RichMediaResponse>.
Question:{question}''' 

class MultimodalRLRAG:
    def __init__(self):
        #初始化模型客户端（使用模拟服务避免真是调用）
        self.client=OpenAI(base_url="http://api.openai.com/v1",api_key="dummy")
        #服务器URL（demo中会被模拟）
        self.search_text_url="http://mock.text.search"
        self.search_visual_url="http://mock.visual.search"
        #图像处理参数
        self.max_pixels=512*28*28
        self.min_pixels=256*28*28
        self.max_steps=10 #限制最大步骤
        
    def process_image(self,image_path):
        """处理图像：调整尺寸并转为base64"""
        #处理输入类型
        if isinstance(image,str) and image.startswith("http"):
            #模拟从URL加载图像
            img=Image.new("RGB",(200,200),color="red") #生成红色测试图
        else:
            img=Image.open(BytesIO(image)) if isinstance(image,bytes) else image
            
        #调整尺寸
        if img.width*img.height>self.max_pixels:
            factor=math.sqrt(self.max_pixels/(img.width*img.height))
            img=img.resize((int(img.width*factor),int(img.height*factor)))
        elif img.width*img.height<self.min_pixels:
            factor=math.sqrt(self.min_pixels/(img.width*img.height))
            img=img.resize((int(img.width*factor),int(img.height*factor)))
        
        #转为RGB并编码
        if img.mode!="RGB":
            img=img.convert("RGB")
        buf=BytesIO()
        img.save(buf,format="JPEG")
        return img,f"data:image;base64,{base64.b64encode(buf.getvalue()).decode()}"
    
    def process_text(self,text):
        """处理文本结果：简单清洗"""
        return [t.strip() for t in text if t.strip()]
    #模拟搜索服务
    def search_text(self,query):
        """模拟文本搜索返回结果"""
        print(f"[模拟文本搜索] 查询：{query}")
        #模拟不同查询的返回结果
        mock_data={
            "苹果":["苹果是一种常见水果，颜色多样，包括红色、绿色等", "苹果富含维生素C"],
            "猫": ["猫是一种哺乳动物，喜欢吃鱼和老鼠", "猫有灵活的爪子和敏锐的听觉"]
        }
        return mock_data.get(query, [f"关于'{query}'的文本信息"])
    
    def search_visual(self,query):
        """模拟图像搜索返回路径"""
        print(f"[模拟图像搜索] 查询：{query}")
        #生成1张测试图像作为搜索结果
        img=Image.new("RGB",(300,300),color="green")
        img_path=f"mock_{query}.jpg"
        img.save(img_path)
        return [img_path]
    
    def calc_reward(self,pred_ans,gold_ans):
        """计算奖励：评估答案质量（简化版）"""
        pred_words=set(pred_ans.lower().split())
        gold_words=set(gold_ans.lower().split())
        #答案准确率作为奖励
        accuracy=len(pred_words & gold_words)/max(len(gold_words),1)
        return round(accuracy*10,2) #奖励范围0～10
    
    def run(self,question,gold_ans=""):
        """主流程：处理问题->调用工具->生成答案->计算奖励"""
        self.visual_history=[] #图像历史
        self.text_history=[]   #文本历史
        message=[{
            "role":"user",
            "content":[{"type":"text","text":prompt_ins.format(question=question)}]
        }]
        
        for step in range(self.max_steps):
            print(f"\n====步骤{step+1}/{self.max_steps}====")
            #调用模型获取相应（模拟多模态模型输出）
            with patch.object(self.client.chat.completions, 'create') as mock_create:
                #模拟模型返回
                if "苹果" in question:
                    mock_response=self._mock_apple_response(step)
                elif "猫" in question:
                    mock_response=self._mock_cat_response(step)
                else:
                    mock_response='现在需要看苹果的图片确认颜色<RichMediaReference><search>红色苹果</search>'
        else:
            return 'superscript:结合文本和图片，苹果是红色的水果<RichMediaReference><answer>这是一个红色的苹果，属于常见水果，富含维生素C</answer>'
    
    #模拟关于猫的多模态响应
    def _mock_cat_response(self,step):
        if step==0:
            return '我需要先了解猫的特征<RichMediaReference><search_text>猫</search_text>'
        elif step==1:
            return '需要看图片确认猫的颜色<RichMediaReference><search>黑猫</search>'
        else:
            return 'superscript:结合信息，这是一只黑色的猫<RichMediaReference><answer>这是一只黑色的猫，属于哺乳动物，有灵活的爪子</answer>'
#主程序入口：测试demo功能
if __name__=="__main__":
    print("===多模态RAG+RL演示程序===")
    #创建智能体实例
    agent=MultimodalRLRAG()
    
    # 测试1：苹果问题
    print("\n----- 测试1：苹果识别 -----")
    question1 = "描述图片中的苹果"
    gold_ans1 = "这是一个红色的苹果，属于常见水果，富含维生素C"
    for result in agent.run(question1, gold_ans1):
        print(f"{result[0]}: {result[1]} {result[2]}")
    
    # 测试2：猫问题
    print("\n----- 测试2：猫识别 -----")
    question2 = "描述图片中的猫"
    gold_ans2 = "这是一只黑色的猫，属于哺乳动物，有灵活的爪子"
    for result in agent.run(question2, gold_ans2):
        print(f"{result[0]}: {result[1]} {result[2]}")
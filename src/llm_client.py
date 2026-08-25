"""
大模型API客户端封装
支持通义千问、文心一言等主流大模型
"""
import json
import time
import requests

class LLMClient:
    """大模型API客户端"""
    
    def __init__(self, api_key, model_provider="qwen", model_name="qwen-turbo"):
        self.api_key = api_key
        self.model_provider = model_provider
        self.model_name = model_name
        self.api_urls = {
            "qwen": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            "ernie": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-3.5-turbo",
        }
    
    def call(self, prompt, system_prompt="你是一个农学数据分析专家。", max_retries=3):
        """调用大模型API，支持重试"""
        for attempt in range(max_retries):
            try:
                if self.model_provider == "qwen":
                    return self._call_qwen(prompt, system_prompt)
                elif self.model_provider == "ernie":
                    return self._call_ernie(prompt, system_prompt)
                else:
                    raise ValueError(f"不支持的模型提供商: {self.model_provider}")
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise Exception(f"大模型API调用失败，已重试{max_retries}次: {str(e)}")
    
    def _call_qwen(self, prompt, system_prompt):
        """调用通义千问API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "input": {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            },
            "parameters": {
                "temperature": 0.1,
                "top_p": 0.8,
                "result_format": "message"
            }
        }
        response = requests.post(self.api_urls["qwen"], headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["output"]["choices"][0]["message"]["content"]
    
    def _call_ernie(self, prompt, system_prompt):
        """调用文心一言API"""
        token_url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={self.api_key}&client_secret={self.api_key}"
        token_resp = requests.get(token_url, timeout=10)
        access_token = token_resp.json().get("access_token", "")
        
        url = f"{self.api_urls['ernie']}?access_token={access_token}"
        payload = {
            "messages": [
                {"role": "user", "content": f"{system_prompt}\n\n{prompt}"}
            ],
            "temperature": 0.1
        }
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["result"]

class MockLLMClient:
    """模拟大模型客户端 - 用于没有API Key时的演示和测试"""
    
    def __init__(self, hallucination_rate=0.15):
        self.hallucination_rate = hallucination_rate
        self.call_count = 0
    
    def call(self, prompt, system_prompt=""):
        """模拟调用，返回预设的结构化数据"""
        self.call_count += 1
        import random
        
        sample_data = [
            {"品种名称": "华粳8号", "种质类型": "选育品种", "来源地": "江苏", "产量(公斤/亩)": 685.5, "株高(cm)": 98, "生育期(天)": 152, "抗稻瘟病": "抗", "抗白叶枯病": "中抗", "蛋白质含量(%)": 8.5, "直链淀粉(%)": 17.2, "胶稠度(mm)": 68},
            {"品种名称": "南粳9108", "种质类型": "选育品种", "来源地": "江苏", "产量(公斤/亩)": 652.3, "株高(cm)": 102, "生育期(天)": 155, "抗稻瘟病": "中抗", "抗白叶枯病": "感", "蛋白质含量(%)": 9.2, "直链淀粉(%)": 14.8, "胶稠度(mm)": 72},
            {"品种名称": "武运粳24", "种质类型": "选育品种", "来源地": "江苏", "产量(公斤/亩)": 678.0, "株高(cm)": 95, "生育期(天)": 150, "抗稻瘟病": "抗", "抗白叶枯病": "中抗", "蛋白质含量(%)": 8.8, "直链淀粉(%)": 16.5, "胶稠度(mm)": 65},
            {"品种名称": "镇稻18号", "种质类型": "选育品种", "来源地": "江苏", "产量(公斤/亩)": 665.2, "株高(cm)": 100, "生育期(天)": 153, "抗稻瘟病": "中抗", "抗白叶枯病": "中抗", "蛋白质含量(%)": 9.0, "直链淀粉(%)": 15.8, "胶稠度(mm)": 70},
            {"品种名称": "徐稻9号", "种质类型": "选育品种", "来源地": "江苏", "产量(公斤/亩)": 645.8, "株高(cm)": 105, "生育期(天)": 158, "抗稻瘟病": "感", "抗白叶枯病": "感", "蛋白质含量(%)": 8.6, "直链淀粉(%)": 18.2, "胶稠度(mm)": 62},
        ]
        
        result = []
        for item in sample_data:
            item_copy = item.copy()
            if random.random() < self.hallucination_rate:
                field = random.choice(["株高(cm)", "产量(公斤/亩)", "抗稻瘟病", "蛋白质含量(%)"])
                if field == "抗稻瘟病":
                    item_copy[field] = random.choice(["高抗", "高感"])
                elif field in ["株高(cm)", "产量(公斤/亩)", "蛋白质含量(%)"]:
                    item_copy[field] = item[field] * random.choice([0.5, 1.5, 2.0])
            result.append(item_copy)
        
        import time
        time.sleep(0.5)
        return json.dumps(result, ensure_ascii=False, indent=2)

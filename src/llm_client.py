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
        self.call_count += 1
        import random

        is_trait = "性状调查" in prompt or "生长习性" in prompt

        if is_trait:
            sample_data = [
                {"编号": "241S21-混红A-1", "生长习性": 1, "裂果性": 2, "果肩": 1, "熟前果色": 3, "萼片形态": 2, "花序类型": 2, "花序长度": 2, "串形": 2, "串形间隔": 2, "熟性": 2, "生长势": 1, "备注病害果尖等": "", "首花节位_1": 8, "首花节位_2": 9, "首花节位_3": 10, "花序间隔节位_1": 3, "花序间隔节位_2": 3, "花序间隔节位_3": 3},
                {"编号": "241S16-2A-1", "生长习性": 1, "裂果性": 2, "果肩": 1, "熟前果色": 3, "萼片形态": 3, "花序类型": 2, "花序长度": 2, "串形": 2, "串形间隔": 2, "熟性": 2, "生长势": 2, "备注病害果尖等": "", "首花节位_1": 7, "首花节位_2": 8, "首花节位_3": 9, "花序间隔节位_1": 3, "花序间隔节位_2": 3, "花序间隔节位_3": 3},
                {"编号": "241S14-1A-1", "生长习性": 1, "裂果性": 1, "果肩": 2, "熟前果色": 3, "萼片形态": 2, "花序类型": 1, "花序长度": 2, "串形": 1, "串形间隔": 2, "熟性": 1, "生长势": 1, "备注病害果尖等": "", "首花节位_1": 9, "首花节位_2": 10, "首花节位_3": 11, "花序间隔节位_1": 3, "花序间隔节位_2": 3, "花序间隔节位_3": 4},
                {"编号": "241H36-1A-1", "生长习性": 1, "裂果性": 2, "果肩": 1, "熟前果色": 2, "萼片形态": 2, "花序类型": 2, "花序长度": 2, "串形": 2, "串形间隔": 2, "熟性": 2, "生长势": 1, "备注病害果尖等": "", "首花节位_1": 8, "首花节位_2": 9, "首花节位_3": 10, "花序间隔节位_1": 3, "花序间隔节位_2": 3, "花序间隔节位_3": 3},
                {"编号": "241FL-29-1G-18", "生长习性": 1, "裂果性": 3, "果肩": 1, "熟前果色": 4, "萼片形态": 3, "花序类型": 2, "花序长度": 2, "串形": 3, "串形间隔": 3, "熟性": 3, "生长势": 1, "备注病害果尖等": "黄曲很严重", "首花节位_1": 10, "首花节位_2": 11, "首花节位_3": 12, "花序间隔节位_1": 4, "花序间隔节位_2": 4, "花序间隔节位_3": 4},
            ]
        else:
            sample_data = [
                {"名称": "241FL-29-1G-5-1", "果重": 10.62, "个数": 2, "单果重": 5.31, "硬度均值": 12.45, "糖度均值": 12.0, "颜色": "红", "形状": "卵圆", "备注": "(失水)", "萼片长度": "短"},
                {"名称": "242WY9-1-1-1", "果重": 5.56, "个数": 1, "单果重": 5.56, "硬度均值": 16.86, "糖度均值": 10.4, "颜色": "红", "形状": "长圆", "备注": "", "萼片长度": "短"},
                {"名称": "241S23-2A-1-2", "果重": 97.46, "个数": 5, "单果重": 19.49, "硬度均值": 9.87, "糖度均值": 8.7, "颜色": "粉", "形状": "扁圆", "备注": "", "萼片长度": "短"},
                {"名称": "CK9", "果重": 99.48, "个数": 5, "单果重": 19.90, "硬度均值": 8.36, "糖度均值": 8.0, "颜色": "粉", "形状": "扁圆", "备注": "", "萼片长度": "短"},
                {"名称": "241H35-1A-3-2", "果重": 52.86, "个数": 3, "单果重": 17.62, "硬度均值": 9.87, "糖度均值": 9.9, "颜色": "红", "形状": "圆", "备注": "", "萼片长度": "短"},
            ]

        result = []
        for item in sample_data:
            item_copy = item.copy()
            if random.random() < self.hallucination_rate:
                if is_trait:
                    field = random.choice(["熟前果色", "首花节位_1", "生长势", "裂果性"])
                    if field in ["熟前果色", "生长势", "裂果性"]:
                        item_copy[field] = random.choice([7, 8, 9])
                    else:
                        item_copy[field] = item[field] * random.choice([0.3, 2.5, 3.0])
                else:
                    field = random.choice(["硬度均值", "果重", "颜色", "糖度均值"])
                    if field == "颜色":
                        item_copy[field] = random.choice(["紫", "黑"])
                    else:
                        item_copy[field] = item[field] * random.choice([0.3, 2.5, 3.0])
            result.append(item_copy)

        import time
        time.sleep(0.5)
        return json.dumps(result, ensure_ascii=False, indent=2)

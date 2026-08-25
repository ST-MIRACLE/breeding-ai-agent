"""
视觉大模型客户端 - 直接读取扫描图片提取数据
基于通义千问VL（Qwen-VL），跳过OCR环节，直接看图提取结构化数据
"""
import base64
import json
import os
import requests


class VisionLLMClient:
    """视觉大模型客户端（通义千问VL）"""

    def __init__(self, api_key, model_name="qwen-vl-max"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    def _encode_image(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def extract_table_from_image(self, image_path, prompt, max_retries=3):
        """读取图片并提取结构化表格数据，支持重试"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        ext = os.path.splitext(image_path)[1].lower().lstrip(".")
        mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "bmp": "bmp", "webp": "webp"}
        mime = mime_map.get(ext, "jpeg")
        base64_image = self._encode_image(image_path)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{mime};base64,{base64_image}"}
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "temperature": 0.1
        }

        import time
        last_error = None
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=300)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    print(f"  - 请求超时/失败，{wait}秒后重试（第{attempt+1}次）...")
                    time.sleep(wait)
        raise Exception(f"视觉大模型调用失败，已重试{max_retries}次: {last_error}")

    def parse_json_output(self, text):
        """从模型输出中提取JSON"""
        import re
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)

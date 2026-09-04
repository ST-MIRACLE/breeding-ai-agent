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
            "temperature": 0.1,
            "max_tokens": 8000
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
        """从模型输出中提取JSON，增强容错"""
        import re
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        # 尝试提取JSON数组
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            json_str = match.group()
        else:
            json_str = text

        # 尝试直接解析
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # 容错修复：处理常见的格式问题
        # 1. 修复值中未转义的引号（如品种名包含引号）
        # 2. 修复缺少逗号的情况
        # 3. 修复多余的换行符

        # 尝试逐行修复并重新组装
        fixed = self._fix_json_syntax(json_str)
        if fixed:
            return fixed

        # 最后尝试：逐个对象提取
        objects = re.findall(r"\{[^{}]+\}", text)
        if objects:
            result = []
            for obj_str in objects:
                try:
                    result.append(json.loads(obj_str))
                except json.JSONDecodeError:
                    fixed_obj = self._fix_json_syntax(obj_str)
                    if fixed_obj and isinstance(fixed_obj, dict):
                        result.append(fixed_obj)
            if result:
                return result

        raise json.JSONDecodeError("无法解析为JSON", text, 0)

    def _fix_json_syntax(self, json_str):
        """尝试修复常见的JSON格式错误"""
        import re

        # 修复缺少逗号的情况（两个 } 或 " 之间缺少逗号）
        fixed = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
        fixed = re.sub(r'\}\s*\n\s*\{', '},\n{', fixed)

        # 修复值中多余的换行
        fixed = re.sub(r':\s*\n\s*"', ': "', fixed)

        try:
            result = json.loads(fixed)
            return result
        except json.JSONDecodeError:
            pass

        # 尝试更激进的修复：逐行解析
        lines = json_str.strip().split('\n')
        cleaned = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 确保每行以逗号结尾（除非是数组开头/结尾）
            if line not in ['[', ']', '{', '}', '},', '},']:
                if not line.endswith(',') and not line.endswith('}') and not line.endswith(']') and not line.endswith('{'):
                    line = line + ','
            cleaned.append(line)

        try:
            return json.loads('\n'.join(cleaned))
        except json.JSONDecodeError:
            return None

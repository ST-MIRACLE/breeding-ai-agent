"""
文献解析模块
从育种科研文献PDF中提取结构化数据
"""
import os
import re
import json


class PaperParser:
    """育种文献解析器"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.prompt_template = self._load_prompt()
    
    def _load_prompt(self):
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "prompts",
            "paper_analysis.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def parse_pdf(self, pdf_path):
        """解析PDF文献"""
        # 提取文本
        text = self._extract_text_from_pdf(pdf_path)
        if not text:
            return {"error": "无法提取PDF文本内容"}
        
        # 调用LLM分析
        return self.analyze_text(text)
    
    def _extract_text_from_pdf(self, pdf_path):
        """从PDF提取文本（需要安装PyPDF2或pdfplumber）"""
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except ImportError:
            # 降级方案：使用PyPDF2
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(pdf_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except ImportError:
                raise ImportError("请安装pdfplumber或PyPDF2: pip install pdfplumber")
    
    def analyze_text(self, text):
        """用LLM分析文献文本"""
        # 加载知识库
        kb_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "knowledge_base",
            "breeding_knowledge.json"
        )
        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
        
        # 截断过长的文本（避免超出token限制）
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...（内容已截断）"
        
        # 构造提示词
        prompt = self.prompt_template.format(
            breeding_terms=json.dumps(kb["breeding_terms"], ensure_ascii=False, indent=2),
            paper_content=text
        )
        
        system_prompt = "你是一位作物育种领域的资深研究员，擅长从科研文献中提取关键实验数据。"
        
        result = self.llm.call(prompt, system_prompt)
        
        # 尝试解析JSON
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"raw_result": result, "note": "输出非标准JSON格式"}
    
    def extract_tables_from_text(self, text):
        """从文本中提取表格数据（备用方法）"""
        # 简单的表格检测：查找包含多个制表符或多个空格分隔的行
        lines = text.split('\n')
        tables = []
        current_table = []
        
        for line in lines:
            # 如果行包含多个数字或制表符，可能是表格行
            if re.search(r'\d+.*\d+.*\d+', line) or '\t' in line:
                current_table.append(line.strip())
            else:
                if len(current_table) >= 3:  # 至少3行才算表格
                    tables.append(current_table)
                current_table = []
        
        if len(current_table) >= 3:
            tables.append(current_table)
        
        return tables

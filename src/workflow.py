"""
育种AI自动化工作流 - 主入口
串联数据处理、大模型调用、校验、输出全流程
"""
import os
import sys
import time
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from llm_client import LLMClient, MockLLMClient
from validator import DataValidator
from paper_parser import PaperParser

class BreedingWorkflow:
    """育种数据处理工作流"""
    
    def __init__(self, api_key=None, model_provider="qwen", model_name="qwen-turbo", use_mock=False):
        if use_mock or api_key is None:
            print("[提示] 使用模拟大模型（演示模式）")
            self.llm = MockLLMClient(hallucination_rate=0.15)
        else:
            self.llm = LLMClient(api_key, model_provider, model_name)
        
        self.validator = DataValidator()
        self.paper_parser = PaperParser(self.llm)
        self.prompt_template = self._load_prompt("data_extraction.txt")
        self.kb = self._load_knowledge_base()
    
    def _load_prompt(self, filename):
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "prompts",
            filename
        )
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def _load_knowledge_base(self):
        kb_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "knowledge_base",
            "breeding_knowledge.json"
        )
        with open(kb_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def process_experiment_data(self, input_file, output_file=None, batch_size=10):
        print("=" * 60)
        print("  育种实验数据AI处理工作流")
        print("=" * 60)
        
        start_time = time.time()
        
        print(f"\n[1/5] 读取原始数据: {input_file}")
        df = self._read_data(input_file)
        print(f"  - 共 {len(df)} 份种质材料，{len(df.columns)} 个字段")
        
        print("\n[2/5] 数据预处理")
        raw_text = df.to_string(index=False)
        print(f"  - 数据文本长度: {len(raw_text)} 字符")
        
        print("\n[3/5] 调用大模型进行数据提取...")
        prompt = self.prompt_template.format(
            field_definitions=json.dumps(self.kb["field_definitions"], ensure_ascii=False, indent=2),
            valid_enums=json.dumps(self.kb["valid_enums"], ensure_ascii=False, indent=2),
            value_ranges=json.dumps(self.kb["value_ranges"], ensure_ascii=False, indent=2),
            raw_data=raw_text
        )
        print(f"  - 提示词长度: {len(prompt)} tokens (约)")
        
        llm_start = time.time()
        llm_output = self.llm.call(prompt, system_prompt="你是一位资深的作物育种科研数据处理专家。")
        llm_time = time.time() - llm_start
        print(f"  - 大模型处理耗时: {llm_time:.1f} 秒")
        
        print("\n[4/5] 知识库约束 + 输出校验")
        validated_data, validation_report = self.validator.validate_and_fix(llm_output, df)
        print(f"  - 校验记录数: {validation_report['total_records']}")
        print(f"  - 发现问题数: {validation_report['issues_found']}")
        print(f"  - 修正字段数: {validation_report['fields_fixed']}")
        
        print("\n[5/5] 输出结构化数据")
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data",
                "output"
            )
            output_file = os.path.join(output_dir, f"{base_name}_structured.xlsx")
        
        result_df = pd.DataFrame(validated_data)
        result_df.to_excel(output_file, index=False)
        print(f"  - 输出文件: {output_file}")
        
        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"  处理完成！总耗时: {total_time:.1f} 秒")
        print(f"  处理效率: {len(df)/total_time*60:.1f} 份/分钟")
        print(f"{'=' * 60}")
        
        return {
            "input_file": input_file,
            "output_file": output_file,
            "total_records": len(df),
            "total_time": round(total_time, 1),
            "llm_time": round(llm_time, 1),
            "records_per_minute": round(len(df) / total_time * 60, 1),
            "validation_report": validation_report,
            "data": validated_data
        }
    
    def process_paper(self, pdf_path, output_file=None):
        print("=" * 60)
        print("  育种文献AI解析工作流")
        print("=" * 60)
        
        start_time = time.time()
        print(f"\n[1/3] 解析文献: {pdf_path}")
        result = self.paper_parser.parse_pdf(pdf_path)
        print("\n[2/3] 结构化提取完成")
        
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data",
                "output"
            )
            output_file = os.path.join(output_dir, f"{base_name}_analysis.json")
        
        print(f"\n[3/3] 输出分析结果")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  - 输出文件: {output_file}")
        
        total_time = time.time() - start_time
        print(f"\n总耗时: {total_time:.1f} 秒")
        return result
    
    def _read_data(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="育种AI自动化工作流")
    parser.add_argument("--input", "-i", required=True, help="输入文件路径")
    parser.add_argument("--output", "-o", help="输出文件路径")
    parser.add_argument("--api-key", help="大模型API Key")
    parser.add_argument("--mock", action="store_true", help="使用模拟模式（无需API Key）")
    parser.add_argument("--mode", choices=["data", "paper"], default="data", help="处理模式")
    args = parser.parse_args()
    
    workflow = BreedingWorkflow(api_key=args.api_key, use_mock=args.mock)
    if args.mode == "data":
        workflow.process_experiment_data(args.input, args.output)
    else:
        workflow.process_paper(args.input, args.output)

if __name__ == "__main__":
    main()

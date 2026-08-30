"""
图片直读工作流 - 扫描图片 → 视觉大模型 → 结构化Excel
跳过OCR环节，视觉大模型直接看图提取数据，无需人工比对
支持果实品质数据和性状调查数据两种类型
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision_client import VisionLLMClient
from validator import DataValidator


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


VISION_PROMPT_TEMPLATE = """你是一位番茄育种科研数据处理专家。请仔细查看这张实验数据扫描图片，逐行精准提取表格中的所有数据。

【字段定义】
{field_definitions}

【提取规则 - 必须严格遵守】
1. 【逐行提取】图片表格中每一行对应一条记录，不得遗漏、不得合并
2. 【均值计算】硬度和糖度如果图片中是多个重复读数，请计算平均值后填入"硬度均值"和"糖度均值"字段
3. 【个数默认】个数列为空时默认填5
4. 【枚举约束】颜色只能使用：红、黄、粉、绿、橙。形状只能使用：卵圆、圆、长圆、桃、扁圆、高圆、梨。萼片长度只能使用：短、中、长
5. 【空值处理】图片中空白或看不清的单元格填写"未提及"
6. 【单果重】由果重除以个数计算得出

【输出格式】
输出标准JSON数组，每个元素是一行数据，键名必须与字段定义完全一致。只输出JSON，不要其他文字。
"""


def load_config():
    """从config.json读取API Key和默认模型"""
    config_path = os.path.join(PROJECT_ROOT, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def detect_data_type(image_path):
    """根据文件名自动判断数据类型"""
    name = os.path.basename(image_path).lower()
    if any(k in name for k in ["trait", "survey", "xingzhuang", "性状"]):
        return "trait_survey"
    return "fruit_quality"


def process_image(image_path, api_key=None, output_file=None, model=None, data_type=None):
    config = load_config()
    if api_key is None:
        api_key = config.get("api_key")
    if model is None:
        model = config.get("default_model", "qwen-vl-max")
    if data_type is None:
        data_type = detect_data_type(image_path)

    type_names = {
        "fruit_quality": "果实品质数据",
        "trait_survey": "性状调查数据"
    }

    print("=" * 60)
    print(f"  番茄实验数据 - 图片直读工作流（{type_names.get(data_type, data_type)}）")
    print("=" * 60)

    if not api_key:
        print("\n错误: 未提供API Key。请用 --api-key 参数，或在 config.json 中配置。")
        return None

    print(f"\n[1/4] 读取扫描图片: {image_path}")
    if not os.path.exists(image_path):
        print(f"  错误: 文件不存在")
        return None

    kb_path = os.path.join(PROJECT_ROOT, "knowledge_base", "breeding_knowledge.json")
    with open(kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)

    if data_type not in kb:
        print(f"  错误: 知识库中没有数据类型 '{data_type}'")
        return None

    prompt = VISION_PROMPT_TEMPLATE.format(
        field_definitions=json.dumps(kb[data_type]["field_definitions"], ensure_ascii=False, indent=2)
    )

    print(f"\n[2/4] 调用视觉大模型（{model}）直接读图...")
    client = VisionLLMClient(api_key, model_name=model)
    raw_output = client.extract_table_from_image(image_path, prompt)
    print("  - 图片识别完成")

    print(f"\n[3/4] 解析结构化数据 + 知识库校验")
    data = client.parse_json_output(raw_output)
    print(f"  - 识别到 {len(data)} 份材料")

    validator = DataValidator()
    validated, report = validator.validate_and_fix(
        json.dumps(data, ensure_ascii=False), data_type=data_type
    )
    print(f"  - 校验发现问题: {report['issues_found']}")
    print(f"  - 自动修正字段: {report['fields_fixed']}")

    print(f"\n[4/4] 输出结构化Excel")
    if output_file is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join(PROJECT_ROOT, "data", "output")
        output_file = os.path.join(output_dir, f"{base}_structured.xlsx")

    df = pd.DataFrame(validated)

    # 性状调查数据附加编码说明列
    if data_type == "trait_survey" and "code_mappings" in kb[data_type]:
        for field, mapping in kb[data_type]["code_mappings"].items():
            if field in df.columns:
                df[f"{field}_说明"] = df[field].map(
                    lambda x: mapping.get(str(x), "")
                )

    df.to_excel(output_file, index=False)
    print(f"  - 输出文件: {output_file}")
    print(f"  - 数据规模: {len(df)} 行 × {len(df.columns)} 列")

    print(f"\n{'=' * 60}")
    print(f"  处理完成！无需人工比对，数据由视觉大模型直接读图提取")
    print(f"{'=' * 60}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图片直读工作流")
    parser.add_argument("--image", "-i", required=True, help="扫描图片路径")
    parser.add_argument("--api-key", help="通义千问API Key（不填则读取config.json）")
    parser.add_argument("--output", "-o", help="输出文件路径")
    parser.add_argument("--model", help="视觉模型名称（不填则读取config.json）")
    parser.add_argument("--type", "-t", choices=["fruit_quality", "trait_survey"],
                        help="数据类型（不填则根据文件名自动判断）")
    args = parser.parse_args()

    process_image(args.image, args.api_key, args.output, args.model, args.type)

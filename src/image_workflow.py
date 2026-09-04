"""
图片直读工作流 - 扫描图片 → 视觉大模型 → 结构化Excel
严格按图片原始排版，不加额外列
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision_client import VisionLLMClient


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_prompt():
    return """你是一位番茄育种科研数据处理专家。请仔细查看这张实验数据扫描图片，逐行逐列精准提取表格中的所有数据。

【表格结构 - 必须严格遵守】
本表格每一行记录包含以下列，必须全部提取、顺序不能乱：
1. 名称：种质材料编号（如R350、T568等）
2. 果重：果实总重量（克）
3. 个数：果实数量
4. 硬度1：硬度第1次测量值
5. 硬度2：硬度第2次测量值
6. 硬度3：硬度第3次测量值
7. 糖度1：糖度第1次测量值
8. 糖度2：糖度第2次测量值
9. 糖度3：糖度第3次测量值
10. 颜色：果实颜色（如红、黄、橙、粉等）
11. 形状：果实形状（如圆、卵圆、高圆、长圆等）
12. 备注：备注信息，无则填"未提及"
13. 萼片长度：短/中/长

【提取规则 - 必须严格遵守】
1. 【逐行提取】图片表格中每一行对应一条记录，不得遗漏、不得合并
2. 【按图原样】硬度、糖度在图片中各有3个测量值（空格分隔），分别填入硬度1、硬度2、硬度3和糖度1、糖度2、糖度3，一个都不能丢、不能取平均、不能合并
3. 【数值原样】数值必须与图片完全一致，小数点、位数不得改动
4. 【空值处理】图片中空白或看不清的单元格填写"未提及"
5. 【输出格式】输出标准JSON数组，每个元素是一行数据，键为上面13个列名，每一行都必须包含全部13个键，键的顺序与上面一致。只输出JSON，不要输出任何其他文字"""


def load_config():
    config_path = os.path.join(PROJECT_ROOT, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def process_image(image_path, api_key=None, output_file=None, model=None):
    config = load_config()
    if api_key is None:
        api_key = config.get("api_key")
    if model is None:
        model = "qwen-vl-max"

    print("=" * 60)
    print("  图片直读工作流")
    print("=" * 60)

    if not api_key:
        print("\n错误: 未提供API Key")
        return None

    print(f"\n[1/3] 读取图片: {os.path.basename(image_path)}")
    if not os.path.exists(image_path):
        print("  错误: 文件不存在")
        return None

    prompt = build_prompt()

    print(f"\n[2/3] 视觉大模型读图（{model}）...")
    t0 = time.time()
    client = VisionLLMClient(api_key, model_name=model)
    raw_output = client.extract_table_from_image(image_path, prompt)
    elapsed = time.time() - t0
    print(f"  - 识别完成，耗时 {elapsed:.0f} 秒")

    data = client.parse_json_output(raw_output)
    print(f"  - 提取 {len(data)} 行 × {len(data[0]) if data else 0} 列")

    print(f"\n[3/3] 输出Excel")
    df = pd.DataFrame(data)

    if output_file is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join(PROJECT_ROOT, "data", "output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{base}_structured.xlsx")

    # 优先xlsx，失败则csv
    try:
        df.to_excel(output_file, index=False)
    except PermissionError:
        output_file = output_file.replace(".xlsx", ".csv")
        df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"  - 输出: {output_file}")
    print(f"  - 列名: {list(df.columns)}")
    print(f"  - 规模: {len(df)} 行 × {len(df.columns)} 列")

    print(f"\n{'=' * 60}")
    print(f"  完成！列名和排版与原始图片一致")
    print(f"{'=' * 60}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图片直读工作流")
    parser.add_argument("--image", "-i", required=True, help="扫描图片路径")
    parser.add_argument("--api-key", help="API Key（不填读config.json）")
    parser.add_argument("--output", "-o", help="输出文件路径")
    parser.add_argument("--model", default="qwen-vl-max", help="模型名")
    args = parser.parse_args()

    process_image(args.image, args.api_key, args.output, args.model)

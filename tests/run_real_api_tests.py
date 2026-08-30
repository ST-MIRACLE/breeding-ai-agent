# -*- coding: utf-8 -*-
"""
真实API准确率测试脚本
使用 qwen-vl-max 视觉大模型直接读取手写品质表图片，
与Excel基准数据逐字段比对，输出真实准确率和耗时统计。

用法:
  python tests/run_real_api_tests.py --api-key YOUR_KEY --image-dir ../accuracy_test/images_upright --ground-truth-dir ../accuracy_test/excel/数据
"""
import os
import sys
import json
import time
import argparse
import re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vision_client import VisionLLMClient
from validator import DataValidator
from image_workflow import compute_means_from_raw

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 视觉提取prompt：直接读取均值字段
VISION_PROMPT = """你是一位番茄育种科研数据处理专家。请仔细查看这张实验数据扫描图片，逐行精准提取表格中的所有数据。

【字段定义】
- 名称：种质材料编号（如251ZJ-26、CK3等）
- 果重：该样品所有果实总重量（克）
- 个数：测量的果实数量，空白时填5
- 单果重：果重除以个数（克）
- 硬度均值：硬度读数的平均值（kg/cm²），如果图片中是多个读数请计算均值，如果已经是均值列直接照抄
- 糖度均值：糖度读数的平均值（Brix%），如果图片中是多个读数请计算均值，如果已经是均值列直接照抄
- 颜色：成熟果实颜色（红/黄/粉/绿/橙）
- 形状：果实形状（卵圆/圆/长圆/扁圆/高圆/桃）
- 备注：异常信息，没有则填空
- 萼片长度：短/中/长

【提取规则】
1. 逐行提取，不得遗漏、不得合并
2. 数值必须与图片完全一致
3. 颜色只使用：红、黄、粉、绿、橙
4. 形状只使用：卵圆、圆、长圆、扁圆、高圆、桃
5. 空白单元格填"未提及"
6. 个数空白时默认填5

【输出格式】
输出标准JSON数组，每个元素是一行数据。只输出JSON，不要其他文字。
"""

# 颜色同义词归一化
COLOR_MAP = {
    '红': '红', '大红': '红', '亮红': '红', '橙红': '红', '橘红': '红', '桔红': '红',
    '粉红': '粉', '粉': '粉', '淡粉': '粉',
    '黄': '黄', '橙黄': '黄', '橙': '橙', '橘黄': '黄', '桔黄': '黄', '橙色': '橙',
    '白': '白', '绿': '绿', '紫': '紫', '棕红': '红', '暗红': '红', '深红': '红',
    '黄加黄': '黄', '黄加黄圆': '黄', '猪红高': '红', '未熟': '绿',
}
SHAPE_MAP = {
    '圆': '圆', '卵圆': '卵圆', '卵': '卵圆', '椭圆': '卵圆', '高圆': '高圆', '高': '高圆',
    '扁圆': '扁圆', '长圆': '长圆', '心形': '心形', '桃': '心形', '桃形': '心形',
    '苹果形': '扁圆', '印圆': '卵圆', '杏形': '卵圆',
}
SEPAL_MAP = {'长': '长', '中': '中', '短': '短', '短中': '短', '中长': '中', '中短': '中', '长中': '长'}


def norm_color(c):
    if not c or str(c).strip() in ('', 'nan', 'None', '未提及'):
        return ''
    return COLOR_MAP.get(str(c).strip(), str(c).strip())


def norm_shape(s):
    if not s or str(s).strip() in ('', 'nan', 'None', '未提及'):
        return ''
    return SHAPE_MAP.get(str(s).strip(), str(s).strip())


def norm_sepal(s):
    if not s or str(s).strip() in ('', 'nan', 'None', '未提及'):
        return ''
    return SEPAL_MAP.get(str(s).strip(), str(s).strip())


def norm_name(n):
    if not n:
        return ''
    n = str(n).strip().replace('（', '(').replace('）', ')')
    return re.sub(r'\s+', '', n)


def load_ground_truth(gt_dir):
    """加载所有品质Excel作为基准"""
    gt = {}
    for root, dirs, files in os.walk(gt_dir):
        for f in files:
            if not f.endswith('.xlsx'):
                continue
            if not any(k in f for k in ['品质', 'ZJ']):
                continue
            fpath = os.path.join(root, f)
            try:
                xl = pd.ExcelFile(fpath)
                for sheet in xl.sheet_names:
                    if any(k in sheet for k in ['精简', 'Sheet1', '251ZJ', '252ZJ']):
                        for hdr in [1, 0]:
                            df = pd.read_excel(fpath, sheet_name=sheet, header=hdr)
                            if '名称' not in df.columns:
                                continue
                            df = df[df['名称'].notna()].copy()
                            df['名称'] = df['名称'].apply(norm_name)
                            for col in ['果重', '个数', '硬度均值', '糖度均值', '单果重']:
                                if col in df.columns:
                                    df[col] = pd.to_numeric(df[col], errors='coerce')
                            for _, r in df.iterrows():
                                nm = r['名称']
                                if nm and nm not in gt:
                                    gt[nm] = r.to_dict()
                            break
            except Exception as e:
                print(f"  警告: 读取 {f}[{sheet}] 失败: {e}")
    print(f"基准库: {len(gt)} 个材料编号")
    return gt


def match_gt(name, gt):
    """匹配基准：全名→短名加前缀"""
    nm = norm_name(name)
    if nm in gt:
        return gt[nm], nm
    if re.match(r'^ZJ[-_]?\d', nm, re.I):
        num = re.sub(r'^ZJ[-_]?', '', nm)
        for prefix in ['251ZJ-', '252ZJ-']:
            cand = prefix + num
            if cand in gt:
                return gt[cand], cand
    return None, nm


def compare_row(pred, gt_row, tol=0.5):
    """逐字段比对，返回(正确数, 总字段数, 不一致列表)"""
    correct = 0
    total = 0
    mm = []
    fields = ['果重', '个数', '硬度均值', '糖度均值', '颜色', '形状', '萼片长度']

    for field in fields:
        total += 1
        pv = pred.get(field)
        gv = gt_row.get(field)

        if field in ['果重', '硬度均值', '糖度均值']:
            try:
                if pv is not None and gv is not None and not pd.isna(gv):
                    if abs(float(pv) - float(gv)) <= tol:
                        correct += 1
                    else:
                        mm.append((field, pv, round(float(gv), 2)))
                else:
                    mm.append((field, pv, gv))
            except (ValueError, TypeError):
                mm.append((field, pv, gv))
        elif field == '个数':
            pv = 5 if pv is None else int(pv)
            gv = 5 if pd.isna(gv) else int(gv)
            if pv == gv:
                correct += 1
            else:
                mm.append((field, pv, gv))
        elif field == '颜色':
            if norm_color(pv) == norm_color(gv) and norm_color(pv):
                correct += 1
            else:
                mm.append((field, pv, gv))
        elif field == '形状':
            if norm_shape(pv) == norm_shape(gv) and norm_shape(pv):
                correct += 1
            else:
                mm.append((field, pv, gv))
        elif field == '萼片长度':
            if norm_sepal(pv) == norm_sepal(gv) and norm_sepal(pv):
                correct += 1
            else:
                mm.append((field, pv, gv))

    return correct, total, mm


def run_test(api_key, image_dir, gt_dir, model='qwen-vl-max'):
    print("=" * 70)
    print("  番茄育种品质表 - 真实API准确率测试")
    print("=" * 70)

    gt = load_ground_truth(gt_dir)
    client = VisionLLMClient(api_key, model_name=model)
    validator = DataValidator()

    images = sorted([f for f in os.listdir(image_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                     and not f.startswith('_')])

    print(f"\n测试图片: {len(images)} 张")
    print(f"模型: {model}")
    print(f"数值容差: ±0.5")
    print("-" * 70)

    all_correct = 0
    all_total = 0
    all_mismatches = []
    timing = []
    field_stats = {f: {'correct': 0, 'total': 0} for f in
                   ['果重', '个数', '硬度均值', '糖度均值', '颜色', '形状', '萼片长度']}
    per_image = []

    for idx, img in enumerate(images, 1):
        img_path = os.path.join(image_dir, img)
        print(f"\n[{idx}/{len(images)}] {img}")

        t0 = time.time()
        try:
            raw_output = client.extract_table_from_image(img_path, VISION_PROMPT)
            elapsed = time.time() - t0
            timing.append(elapsed)
            print(f"  API耗时: {elapsed:.1f}秒")
        except Exception as e:
            print(f"  API调用失败: {e}")
            timing.append(None)
            continue

        try:
            data = client.parse_json_output(raw_output)
        except Exception as e:
            print(f"  JSON解析失败: {e}")
            continue

        if not isinstance(data, list):
            data = [data]
        print(f"  提取行数: {len(data)}")

        # validator校验
        try:
            validated, report = validator.validate_and_fix(
                json.dumps(data, ensure_ascii=False), data_type="fruit_quality"
            )
            print(f"  校验修正: {report['fields_fixed']} 个字段")
        except Exception as e:
            print(f"  校验跳过: {e}")
            validated = data

        img_correct = 0
        img_total = 0
        img_matched = 0
        img_mm = []

        for row in validated:
            name = row.get('名称', '')
            gt_row, matched_name = match_gt(name, gt)
            if gt_row is None:
                continue
            img_matched += 1
            c, t, mm = compare_row(row, gt_row)
            img_correct += c
            img_total += t
            for field, pv, gv in mm:
                field_stats[field]['total'] += 1
                img_mm.append((matched_name, field, pv, gv))
            for field in ['果重', '个数', '硬度均值', '糖度均值', '颜色', '形状', '萼片长度']:
                field_stats[field]['total'] += 1
                if not any(m[1] == field for m in mm):
                    field_stats[field]['correct'] += 1

        all_correct += img_correct
        all_total += img_total
        all_mismatches.extend(img_mm)

        acc = img_correct / img_total * 100 if img_total > 0 else 0
        per_image.append({'image': img, 'matched': img_matched, 'total_rows': len(data),
                          'correct': img_correct, 'total_fields': img_total,
                          'accuracy': round(acc, 1), 'time_sec': round(timing[-1], 1) if timing[-1] else None})
        print(f"  配对: {img_matched}/{len(data)} 行, 准确率: {img_correct}/{img_total} = {acc:.1f}%")
        for nm, field, pv, gv in img_mm[:3]:
            print(f"    {nm} | {field}: 提取={pv} 基准={gv}")
        if len(img_mm) > 3:
            print(f"    ...共{len(img_mm)}处不一致")

    # 汇总
    print("\n" + "=" * 70)
    print("  测试结果汇总")
    print("=" * 70)

    valid_timing = [t for t in timing if t is not None]
    if valid_timing:
        print(f"\n耗时统计: 平均{np.mean(valid_timing):.1f}秒/张, "
              f"最快{min(valid_timing):.1f}秒, 最慢{max(valid_timing):.1f}秒")

    overall_acc = all_correct / all_total * 100 if all_total > 0 else 0
    print(f"\n总体准确率: {all_correct}/{all_total} = {overall_acc:.1f}%")
    print(f"测试图片: {len(images)} 张, 成功: {len(valid_timing)} 张")

    print(f"\n各字段准确率:")
    for field, stats in field_stats.items():
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            print(f"  {field}: {stats['correct']}/{stats['total']} = {acc:.1f}%")

    # 保存结果
    output_dir = os.path.join(PROJECT_ROOT, "data", "output")
    os.makedirs(output_dir, exist_ok=True)
    result = {
        'model': model,
        'tolerance': 0.5,
        'total_images': len(images),
        'successful_images': len(valid_timing),
        'avg_time_sec': round(np.mean(valid_timing), 1) if valid_timing else None,
        'overall_accuracy': round(overall_acc, 1),
        'total_fields': all_total,
        'correct_fields': all_correct,
        'per_field_accuracy': {f: round(s['correct'] / s['total'] * 100, 1) if s['total'] else 0
                                for f, s in field_stats.items()},
        'per_image': per_image,
        'mismatches': [{'name': m[0], 'field': m[1], 'predicted': m[2], 'ground_truth': m[3]}
                       for m in all_mismatches]
    }
    result_file = os.path.join(output_dir, "real_api_accuracy_result.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {result_file}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="真实API准确率测试")
    parser.add_argument("--api-key", required=True, help="DashScope API Key")
    parser.add_argument("--image-dir", required=True, help="测试图片目录")
    parser.add_argument("--ground-truth-dir", required=True, help="基准Excel目录")
    parser.add_argument("--model", default="qwen-vl-max", help="视觉模型名称")
    args = parser.parse_args()
    run_test(args.api_key, args.image_dir, args.ground_truth_dir, args.model)

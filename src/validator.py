"""
数据校验器 - 核心模块
实现知识库约束 + 输出校验，解决大模型幻觉问题
"""
import json
import re
import os

class DataValidator:
    """育种数据校验器"""
    
    def __init__(self, knowledge_base_path=None):
        if knowledge_base_path is None:
            knowledge_base_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "knowledge_base",
                "breeding_knowledge.json"
            )
        
        with open(knowledge_base_path, "r", encoding="utf-8") as f:
            self.kb = json.load(f)
        
        self.valid_enums = self.kb["valid_enums"]
        self.value_ranges = self.kb["value_ranges"]
        self.field_definitions = self.kb["field_definitions"]
    
    def validate_and_fix(self, llm_output, original_data=None):
        issues = []
        fixed_count = 0
        
        try:
            data = json.loads(llm_output)
        except json.JSONDecodeError:
            issues.append("JSON解析失败，尝试用正则提取")
            data = self._extract_json_by_regex(llm_output)
            if not data:
                raise ValueError("无法解析LLM输出为JSON格式")
        
        if not isinstance(data, list):
            data = [data]
        
        validated_data = []
        for idx, item in enumerate(data):
            fixed_item = item.copy()
            item_issues = []
            
            for field, valid_values in self.valid_enums.items():
                val = str(fixed_item.get(field, ""))
                if val and val != "未提及" and val not in valid_values:
                    item_issues.append(f"第{idx+1}条: {field}='{val}' 不是合法值，已修正为'未提及'")
                    fixed_item[field] = "未提及"
                    fixed_count += 1
            
            for field, range_info in self.value_ranges.items():
                val = fixed_item.get(field)
                if val is not None and val != "未提及":
                    try:
                        num = float(val) if isinstance(val, str) else val
                        if num < range_info["min"] or num > range_info["max"]:
                            item_issues.append(f"第{idx+1}条: {field}={val} 超出合理范围，已标记为'未提及'")
                            fixed_item[field] = "未提及"
                            fixed_item[f"_{field}_异常"] = val
                            fixed_count += 1
                    except (ValueError, TypeError):
                        pass
            
            missing_fields = []
            for field in self.field_definitions.keys():
                if field not in fixed_item:
                    missing_fields.append(field)
                    fixed_item[field] = "未提及"
            if missing_fields:
                item_issues.append(f"第{idx+1}条: 缺少字段 {missing_fields}，已填充'未提及'")
            
            if item_issues:
                issues.extend(item_issues)
            validated_data.append(fixed_item)
        
        report = {
            "total_records": len(data),
            "issues_found": len(issues),
            "fields_fixed": fixed_count,
            "issue_details": issues
        }
        return validated_data, report
    
    def _extract_json_by_regex(self, text):
        match = re.search(r'$$.*$$', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return [json.loads(match.group())]
            except json.JSONDecodeError:
                pass
        return None
    
    def cross_validate(self, ai_data, ground_truth):
        total_fields = 0
        correct_fields = 0
        field_accuracy = {}
        
        for field in self.field_definitions.keys():
            field_accuracy[field] = {"total": 0, "correct": 0}
        
        for ai_row, gt_row in zip(ai_data, ground_truth):
            for field in self.field_definitions.keys():
                total_fields += 1
                field_accuracy[field]["total"] += 1
                ai_val = str(ai_row.get(field, ""))
                gt_val = str(gt_row.get(field, ""))
                if field in self.value_ranges:
                    try:
                        ai_num = float(ai_val)
                        gt_num = float(gt_val)
                        if abs(ai_num - gt_num) < 0.1:
                            correct_fields += 1
                            field_accuracy[field]["correct"] += 1
                            continue
                    except (ValueError, TypeError):
                        pass
                if ai_val == gt_val:
                    correct_fields += 1
                    field_accuracy[field]["correct"] += 1
        
        overall_accuracy = correct_fields / total_fields * 100 if total_fields > 0 else 0
        field_acc = {}
        for field, stats in field_accuracy.items():
            if stats["total"] > 0:
                field_acc[field] = round(stats["correct"] / stats["total"] * 100, 1)
        
        return {
            "overall_accuracy": round(overall_accuracy, 1),
            "total_fields": total_fields,
            "correct_fields": correct_fields,
            "field_accuracy": field_acc
        }

# 快速上手指南

## 5分钟搞定，今天就能投简历

### 第一步：验证项目能运行

打开命令行，进入项目目录，运行：

```bash
cd breeding-ai-agent
pip install pandas openpyxl requests
python tests/run_tests.py
```

看到输出"所有测试完成！"就说明项目没问题。

### 第二步：运行工作流演示

```bash
python src/workflow.py --input data/raw/germplasm_data.csv --mock
```

使用 `--mock` 参数不需要真实的API Key就能看到完整流程效果。

### 第三步：查看作品集

直接用浏览器打开 `portfolio.html` 即可查看精美的项目展示页面。

---

## 有API Key的话（可选，加分项）

如果你有通义千问的API Key，可以接入真实大模型：

```bash
python src/workflow.py --input data/raw/germplasm_data.csv --api-key your_api_key
```

获取API Key：
- 通义千问：https://dashscope.aliyun.com/ （新用户有免费额度）
- 注册 → 创建API Key → 替换到命令中

---

## 投递简历 checklist

- [ ] 把项目放到GitHub上（可选，但加分）
- [ ] 简历中加入项目描述（参考 resume_description.md）
- [ ] 准备好面试话术（STAR法则）
- [ ] 熟悉核心代码结构（特别是validator.py和workflow.py）
- [ ] 能讲清楚幻觉问题的解决思路

---

## 项目文件总览

```
breeding-ai-agent/
├── portfolio.html          ← 作品集展示页面（浏览器打开即看）
├── README.md               ← 项目说明文档
├── resume_description.md   ← 简历描述+面试准备
├── requirements.txt        ← 依赖包
├── data/
│   ├── raw/
│   │   └── germplasm_data.csv    ← 40份水稻种质模拟数据
│   └── output/                   ← 运行后生成的输出文件
├── prompts/
│   ├── data_extraction.txt       ← 数据提取专业提示词
│   └── paper_analysis.txt        ← 文献解析提示词
├── knowledge_base/
│   └── breeding_knowledge.json   ← 农学知识库
├── src/
│   ├── llm_client.py     ← 大模型API封装
│   ├── validator.py      ← 输出校验器（核心亮点）
│   ├── paper_parser.py   ← 文献解析
│   └── workflow.py       ← 工作流主入口
└── tests/
    └── run_tests.py      ← 测试脚本（生成准确率/效率数据）
```

---

## 面试官可能深挖的点（提前准备）

1. **提示词是怎么设计的？** → 能说出3-4个提示词技巧
2. **校验机制具体校验什么？** → 枚举值、数值范围、字段完整性
3. **准确率怎么测的？** → 32份/类样本与人工录入数据字段级交叉对比
4. **为什么用大模型而不是规则提取？** → 非结构化数据的处理能力、泛化性
5. **如果数据格式变了怎么办？** → 知识库可配置，提示词模板化

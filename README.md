# 智能规划数据生成工具链

本仓库提供一条端到端的数据生成流水线：

1)  从对话数据中自动挖掘 任务场景 → 工具集 映射关系\
2)  基于场景-工具集配置，自动合成用于偏好训练的规划数据

------------------------------------------------------------------------

# 一、Scene → Toolset 构建

## 功能说明

-   解析对话中的 `Action: tool_name`
-   聚合 service（根据 `_for_` 后缀）
-   使用 sentence-transformers 生成工具 embedding
-   使用 HDBSCAN 在同一 service 内进行语义聚类
-   输出更细粒度的 "主题(theme) → 工具集(tools)"

## 运行方式

python tools-cluster/cluster_tool.py

## 输出文件

-   toolbench_semantic_compounds.json
-   test_small.json
-   test_large.json
-   sample_semantic.json

------------------------------------------------------------------------

# 二、Preference 数据合成

## 环境变量

在根目录创建 `.env` 文件：

OPENAI_API_KEY=your_key\
OPENAI_BASE_URL=your_base_url

支持任意兼容 OpenAI Chat Completions API 的模型服务。

## 运行方式

python preference-data-synthesis/generate_preference_data.py --input
test_small.json --output preference_data.json --num_plans 8

## 生成流程

1.  生成真实用户问题
2.  生成多个规划 Plan（2-4 步）
3.  规则打分选优
4.  主动降级构造负样本
5.  LLM judge 校验
6.  输出 prompt / chosen / rejected

## 输出数据结构

{ "prompt": "...", "chosen": {...}, "rejected": {...}, "task_theme":
"...", "generate_time": "..." }

------------------------------------------------------------------------

# 设计核心思想

-   Service 内语义聚类 → 更细粒度工具子域
-   主动降级负样本 → 提高偏好数据稳定性
-   规则评分 + LLM judge 双保险

------------------------------------------------------------------------

# License

Apache-2.0

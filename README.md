# 智能规划数据生成工具链

本仓库包含两条核心流水线：

1)  **场景-工具集构建（Scene → Toolset）**\
    从对话数据中自动抽取工具链，按 service 聚合后，用句向量 + HDBSCAN
    对同一 service 内工具做语义聚类，得到更细粒度的 **"主题(theme) →
    工具集(tools)"** 任务配置（compounds）。

2)  **偏好数据合成（Preference Data Synthesis）**\
    基于上游产出的 theme/tools 任务配置，调用 LLM：

-   生成真实用户问题（不暴露工具/API）
-   生成多样化规划 Plan（含 thought / tools / dependencies）
-   规则打分筛选高质量方案
-   主动降级构造强负样本
-   可选 LLM judge 校验
-   输出 pairwise 偏好训练数据（prompt / chosen / rejected）

------------------------------------------------------------------------

## 代码结构

-   `cluster_tool.py`\
    解析对话 → service 聚合 → 语义聚类 → 输出 compounds 与 test 任务文件

-   `generate_preference_data.py`\
    生成问题 → 多 Plan 生成 → 规则评分 → 主动降级负样本 → judge 校验 →
    输出偏好数据

------------------------------------------------------------------------

## 安装

建议 Python 3.10+

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 一、Scene → Toolset 构建

## 数据来源

默认使用 HuggingFace 数据集：

    Yhyu13/ToolBench_toolllama_G123_dfs (split=train)

脚本从 assistant 消息中解析：

    Action: tool_name

形成每个场景的工具链（去重）。

## 运行

``` bash
python cluster_tool.py
```

## 关键参数

在 `cluster_tool.py` 顶部可修改：

-   `MIN_TOOLS_PER_SCENE`
-   `MIN_TOOLS_PER_SERVICE`
-   `MIN_SERVICE_SUPPORT`
-   `HDBSCAN_MIN_CLUSTER_SIZE`
-   `HDBSCAN_MIN_SAMPLES`
-   `EMBED_MODEL`

## 输出文件

-   `toolbench_semantic_compounds.json`
-   `test_small.json`
-   `test_large.json`
-   `sample_semantic.json`

------------------------------------------------------------------------

# 二、偏好数据生成

## 环境变量

在仓库根目录创建 `.env` 文件：

    OPENAI_API_KEY=你的key
    OPENAI_BASE_URL=你的base_url

支持任意兼容 OpenAI Chat Completions 接口的服务。

## 运行

``` bash
python generate_preference_data.py \
  --input test_small.json \
  --output preference_data.json
```

常用参数：

-   `--num_plans` 每个任务生成的候选 Plan 数量
-   `--model` 模型名称
-   `--temperature` Plan 生成温度
-   `--no_degrade` 关闭主动降级模式

------------------------------------------------------------------------

## 输出数据格式

每条偏好样本：

``` json
{
  "prompt": "...",
  "chosen": { ... },
  "rejected": { ... },
  "task_theme": "...",
  "generate_time": "..."
}
```

默认使用"主动降级"构造高质量负样本。

------------------------------------------------------------------------

## 设计理念

### 1. Service 内语义聚类

同一平台工具语义相近但功能不同， 通过 embedding + HDBSCAN
获得更细粒度工具子域。

### 2. 主动降级负样本

通过构造明显错误的规划（工具错误 / 步骤缺失 / 内容模糊），
提高偏好数据质量稳定性。

------------------------------------------------------------------------

## License
Apache-2.0

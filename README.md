# 智能规划数据生成工具链

基于 ToolBench 构建**工具调用偏好数据集**的数据工程工具包。

整体流程分为两个阶段：先对 API 工具按语义相似度进行聚类，再通过 LLM 生成用户问题与多步骤规划，利用主动降级策略合成高质量的 `(chosen, rejected)` 偏好对。

---

## 流程概览

```
ToolBench 数据集
      │
      ▼
tools-cluster/cluster_semantic.py     # 阶段一：工具语义聚类
      │  输出：toolbench_semantic_compounds.json
      │        test_small.json（每簇 ≤10 个工具）
      │        test_large.json（每簇 >10 个工具）
      ▼
preference-data-synthesis/generate_preference_data.py      # 阶段二：生成偏好数据对
      │  输出：test/preference_data.json
      ▼
   训练数据
```

---

## 项目结构

```

├── tools-cluster/
│   ├── cluster_semantic.py               # 工具语义聚类脚本
│   ├── sample_semantic.json              # 聚类结果样例（Top 5）
│   ├── test_small.json                   # ≤10 个工具的簇（推荐用于生成）
│   ├── test_large.json                   # >10 个工具的簇
│   └── toolbench_semantic_compounds.json # 完整聚类结果
├── preference-data-synthesis/
│   ├── generate_preference_data.py       # 偏好数据生成脚本
│   ├── test/
│   │   ├── test.json                        # 输入：工具簇配置
│   └── └── preference_data.json             # 输出：最终数据集
│   
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install datasets sentence-transformers hdbscan openai tenacity python-dotenv
```

---

## 环境配置

在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=你的 API Key
OPENAI_BASE_URL=https://你的接口地址/v1
```

---

## 快速开始

```bash
# 阶段一：工具语义聚类
cd tools-cluster
python cluster_semantic.py

# 阶段二：生成偏好数据
cd ../test
python generate_preference_data.py --input test.json --output preference_data.json
```

第二个脚本支持**断点续传**，中断后重新运行同一命令，会自动从 `resume_preference.json` 恢复进度。

---

## 脚本说明

### `tools-cluster/cluster_semantic.py` — 工具语义聚类

从 ToolBench 数据集中解析工具调用链，按语义相似度将同一服务下的工具聚成若干功能簇。

**处理流程：**
1. 从对话轨迹的 assistant 轮次中提取工具调用序列
2. 按 `_for_` 后缀将工具归属到对应服务
3. 用 `sentence-transformers` 对工具名进行语义编码
4. 在每个服务内部用 HDBSCAN 进行聚类
5. 对大簇或聚类失败的情况，降级为关键词切分
6. 输出带主题描述和场景支持数的工具复合体

**关键参数**（位于文件顶部）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `MIN_TOOLS_PER_SCENE` | 2 | 每个场景最少工具数 |
| `MIN_TOOLS_PER_SERVICE` | 3 | 服务至少包含的工具数 |
| `MIN_SERVICE_SUPPORT` | 5 | 服务至少出现的场景数 |
| `HDBSCAN_MIN_CLUSTER_SIZE` | 2 | 簇最少工具数 |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | 语义编码模型 |

---

### `test/generate_preference_data.py` — 偏好数据生成

读取工具簇配置，通过 LLM 生成用户问题和多步骤规划，构造训练所需的偏好数据对。

**处理流程：**
1. 用 LLM 过滤出与主题相关的工具子集
2. 基于主题生成真实用户问题
3. 用多样性提示生成 N 个候选规划
4. 多维度规则评分，取最高分规划作为 `chosen`
5. 通过**主动降级**策略构造 `rejected`
6. 用 LLM 裁判验证偏好对有效性（含 AB 位置偏差修正）
7. 断点续传保存结果

**`rejected` 构造策略：**

| 策略 | 说明 |
|---|---|
| `all_wrong_tools` | 将所有工具调用替换为不存在的工具 |
| `missing_step` | 删除所有中间步骤，仅保留首尾 |
| `single_step` | 将整个规划压缩为一个模糊步骤 |
| `vague_content` | 将所有步骤内容替换为无意义的模糊描述 |
| `wrong_tools_and_vague` | 组合降级：工具错误 + 内容模糊 |

**规则评分维度：**

| 维度 | 权重 | 说明 |
|---|---|---|
| 工具调用合理性 | 0.35 | 调用工具均在候选池内，无幻觉工具 |
| 步骤逻辑性 | 0.25 | 2–4 步最优，3 步得满分 |
| 内容具体性 | 0.20 | 包含具体操作，无模糊描述 |
| 依赖关系合理性 | 0.10 | 步骤间依赖关系是否合理设置 |
| 关键词匹配度 | 0.10 | 匹配核心业务/工具关键词 |

**命令行参数：**

```bash
python generate_preference_data.py \
  --input test.json \             # 输入工具簇文件
  --output preference_data.json \ # 输出文件
  --num_plans 8 \                 # 每个任务生成的候选规划数
  --model qwen3-coder-next \      # 使用的模型
  --temperature 0.9 \             # 规划生成温度
  --no_degrade                    # 可选：禁用主动降级，改用 LLM 对比模式
```

---

## 输出格式

`preference_data.json` 中每条记录的结构：

```json
{
  "prompt": "用户提出的业务问题",
  "chosen": {
    "fixed_question": "标准化后的问题描述",
    "thought": "整体规划思路",
    "steps": [
      {
        "thought": "本步骤的推理过程",
        "title": "步骤标题",
        "content": "具体执行内容",
        "tools": ["tool_name_for_service"],
        "dependencies": null
      }
    ]
  },
  "rejected": { "..." : "..." },
  "degrade_strategy": "all_wrong_tools",
  "task_theme": "search finance",
  "generate_time": "2025-01-01 12:00:00"
}
```

---

## 注意事项

- LLM 裁判使用**随机 AB 位置交换**来消除位置偏差
- 降级后的 `rejected` 若裁判意外判为优胜，该对数据会被自动丢弃，避免污染训练集
- `FALLBACK_FUNC_GROUPS` 中包含金融、体育、社交平台等领域的关键词分组，如需支持新领域，在此扩展即可

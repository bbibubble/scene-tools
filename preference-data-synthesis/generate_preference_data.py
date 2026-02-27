import os
import json
import time
import random
import argparse
import traceback
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()


# ===================== 配置类 =====================
@dataclass
class Config:
    input_file: str = "test.json"
    output_file: str = "preference_data.json"
    resume_file: str = "resume_preference.json"
    num_plans: int = 8
    top_k: int = 1          # 每个prompt只取最优Plan，避免同质化样本
    model: str = "qwen3-coder-next"
    temperature_question: float = 0.8
    temperature_plan: float = 0.9
    temperature_judge: float = 0.0
    api_delay: float = 0.5
    max_retries: int = 3
    min_plan_length: int = 200
    # 是否使用主动降级构造负样本（推荐True）
    use_degrade: bool = True


# ===================== 参数解析 =====================
def build_config() -> Config:
    parser = argparse.ArgumentParser(description="生成偏好数据（含主动降级负样本）")
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--num_plans", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--no_degrade", action="store_true", help="禁用主动降级，回退到原始对比模式")
    args = parser.parse_args()

    config = Config()
    if args.input:      config.input_file = args.input
    if args.output:     config.output_file = args.output
    if args.num_plans:  config.num_plans = args.num_plans
    if args.model:      config.model = args.model
    if args.temperature: config.temperature_plan = args.temperature
    if args.no_degrade: config.use_degrade = False
    return config


config = build_config()

# ===================== LLM客户端 =====================
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
if not client.api_key or not client.base_url:
    raise ValueError("请配置 OPENAI_API_KEY 和 OPENAI_BASE_URL 环境变量！")


# ===================== 工具函数 =====================
@retry(
    stop=stop_after_attempt(config.max_retries),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: print(f"⚠️  API调用失败，重试 {retry_state.attempt_number}/{config.max_retries}...")
)
def call_llm(prompt: str, temperature: float = 0.9) -> Optional[str]:
    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2048,
            top_p=0.9,
            frequency_penalty=0.1
        )
        result = response.choices[0].message.content.strip()
        if not result:
            raise ValueError("LLM返回空内容")
        return result
    except Exception as e:
        print(f"❌ LLM调用异常: {str(e)[:200]}")
        raise


def validate_json(text: str) -> Optional[Dict]:
    """解析JSON，兼容```json代码块格式"""
    if not text:
        return None
    # 去掉 ```json ... ``` 包裹
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # 兜底：提取第一个 {...} 块
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        print(f"❌ JSON解析失败: {cleaned[:100]}...")
        return None


def load_resume_data() -> Tuple[List[Dict], int]:
    if os.path.exists(config.resume_file):
        try:
            with open(config.resume_file, "r", encoding="utf-8") as f:
                resume_data = json.load(f)
            preference_data = resume_data.get("data", [])
            task_idx = resume_data.get("task_idx", 0)
            print(f"✅ 加载断点数据：已生成 {len(preference_data)} 条样本，当前任务索引 {task_idx}")
            return preference_data, task_idx
        except Exception as e:
            print(f"⚠️  加载断点失败，重新开始：{e}")
    return [], 0


def save_resume_data(preference_data: List[Dict], task_idx: int):
    try:
        resume_data = {"data": preference_data, "task_idx": task_idx, "config": asdict(config)}
        with open(config.resume_file, "w", encoding="utf-8") as f:
            json.dump(resume_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️  保存断点失败：{e}")


# ===================== 核心生成逻辑 =====================
def generate_question(theme: str, tools: List[str]) -> Optional[str]:
    prompt = f"""
你是一个真实的业务用户，需要基于以下主题提出一个具体的业务问题。

### 要求
1. 问题必须和主题强相关：{theme}
2. 问题要具体，包含明确的业务目标（如地点、时间范围、数据格式等）
3. 问题必须用自然语言表达，不能出现任何工具名称或API名称
4. 问题应该是用户真实会问的，而非技术文档描述
5. 输出格式：仅输出问题文本，无其他内容

### 主题
{theme}
"""
    return call_llm(prompt, temperature=config.temperature_question)


def filter_tools_by_theme(theme: str, tools: List[str]) -> List[str]:
    """用LLM从全量工具列表中筛选出与主题相关的工具，避免跨平台工具干扰"""
    if len(tools) <= 5:
        return tools  # 工具数量少时直接返回，不必过滤

    prompt = f"""
你是一个工具选择专家。给定一个任务主题和一批工具，请从中筛选出与主题**直接相关**的工具。

### 判断标准
1. 工具名称或功能与主题关键词有明确关联
2. 工具属于同一平台/服务（工具名通常包含 _for_平台名 后缀，优先选同平台工具）
3. 宁可少选也不要选无关工具，数量不限，按实际相关性决定

### 任务主题
{theme}

### 全量工具列表
{json.dumps(tools, ensure_ascii=False)}

### 输出要求
仅输出JSON数组，无其他内容，示例：["tool_a", "tool_b"]
"""
    try:
        result = call_llm(prompt, temperature=0.0)
        # 兼容 ```json [...] ``` 格式
        cleaned = re.sub(r"```(?:json)?\s*", "", result).strip().rstrip("`").strip()
        filtered = json.loads(cleaned)
        if isinstance(filtered, list) and len(filtered) > 0:
            # 二次校验：过滤掉不在原始列表中的幻觉工具
            valid = [t for t in filtered if t in tools]
            if valid:
                print(f"  🔧 工具过滤：{len(tools)} → {len(valid)} 个（{', '.join(valid)}）")
                return valid
    except Exception as e:
        print(f"  ⚠️  工具过滤失败，使用全量工具：{e}")
    return tools


def generate_plan(question: str, tools: List[str], diversity_hint: str = "") -> Optional[str]:
    """生成Plan，输出含thought和dependencies的新格式"""
    prompt = f"""
你是一个任务规划专家，需要为用户问题生成可执行的多步骤规划。

### 用户问题
{question}

### 可用工具
{', '.join(tools)}

### 生成要求
1. 步骤数量：2-4个
2. 每个步骤包含 thought（推理过程）、title、content、tools、dependencies 五个字段
3. thought：说明为什么这样规划该步骤，工具选择依据，依赖关系判断
4. tools：从可用工具中选择，若该步骤无需工具则为 null
5. dependencies：若依赖前序步骤则填写步骤title列表，否则为 null
6. 顶层需包含 thought（整体规划思路）和 fixed_question（对用户问题的标准化描述）
{diversity_hint}
7. 输出格式：仅输出严格的JSON，无其他内容：
{{
  "fixed_question": "标准化后的用户问题描述",
  "thought": "整体规划思路：分析任务、拆解步骤、判断依赖关系",
  "steps": [
    {{
      "thought": "本步骤的推理：目标、工具选择依据、依赖判断",
      "title": "步骤标题",
      "content": "步骤具体执行内容",
      "tools": ["工具名1"],
      "dependencies": null
    }},
    {{
      "thought": "本步骤的推理",
      "title": "步骤标题2",
      "content": "步骤具体执行内容",
      "tools": null,
      "dependencies": ["步骤标题1"]
    }}
  ]
}}
"""
    plan_text = call_llm(prompt, temperature=config.temperature_plan)
    if validate_json(plan_text):
        return plan_text
    print(f"❌ Plan格式无效，重新生成...")
    return None


# ===================== 主动降级构造负样本 =====================
DEGRADE_STRATEGIES = ["all_wrong_tools", "missing_step", "single_step", "vague_content", "wrong_tools_and_vague"]

def degrade_plan(plan_json: dict, tools: list) -> Tuple[dict, str]:
    """
    对高质量Plan做强力降级，确保缺陷足够明显，judge不会误判。
    返回 (降级后的plan_json, 缺陷说明)
    """
    import copy
    degraded = copy.deepcopy(plan_json)
    steps = degraded.get("steps", [])

    available = list(DEGRADE_STRATEGIES)
    if len(steps) < 3:
        available.remove("missing_step")

    strategy = random.choice(available)

    if strategy == "all_wrong_tools":
        # 只替换原本有工具的步骤，保留tools:[]的纯逻辑步骤不动
        fake_tools = ["invalid_tool_alpha", "invalid_tool_beta", "invalid_tool_gamma"]
        replaced = 0
        for i, step in enumerate(steps):
            if step.get("tools"):   # 只改有工具的步骤
                step["tools"] = [fake_tools[replaced % len(fake_tools)]]
                replaced += 1
        if replaced == 0:
            # 所有步骤都没工具，退化为vague_content
            for step in steps:
                step["content"] = "根据需要执行相关操作，获取所需信息。"
            reason = "步骤内容被替换为模糊描述，缺乏具体执行细节"
        else:
            reason = f"共{replaced}个步骤的工具被替换为不存在的工具，工具调用完全错误"

    elif strategy == "missing_step":
        # 删除中间所有步骤，只保留首尾
        if len(steps) >= 3:
            removed_titles = [s.get("title", "未知") for s in steps[1:-1]]
            degraded["steps"] = [steps[0], steps[-1]]
            reason = f"删除了中间所有关键步骤（{', '.join(removed_titles)}），流程严重不完整"
        else:
            mid = steps.pop(1)
            reason = f"删除了关键步骤'{mid.get('title', '未知')}'"

    elif strategy == "single_step":
        # 压缩为单步骤，且内容模糊；同时替换顶层thought避免逻辑矛盾
        degraded["steps"] = [{
            "thought": "直接执行即可。",
            "title": "执行所有操作",
            "content": "一次性调用所有工具完成任务。",
            "tools": steps[0].get("tools") if steps else None,
            "dependencies": None
        }]
        degraded["thought"] = "直接一步完成所有操作。"
        degraded["fixed_question"] = degraded.get("fixed_question", "")
        reason = "Plan被压缩为单一模糊步骤，缺乏必要的拆分和具体执行逻辑"

    elif strategy == "vague_content":
        # 所有步骤内容和标题全部替换为极度模糊的描述
        for i, step in enumerate(steps):
            step["title"] = f"步骤{i+1}"
            step["thought"] = "执行操作。"
            step["content"] = "执行相关操作。"
            step["tools"] = None
            step["dependencies"] = None
        degraded["thought"] = "按顺序执行各步骤完成任务。"
        reason = "所有步骤标题、内容均被替换为无意义模糊描述，且工具调用缺失"

    elif strategy == "wrong_tools_and_vague":
        # 组合降级：工具错误 + 内容模糊（双重缺陷）
        for i, step in enumerate(steps):
            step["tools"] = ["wrong_tool_xyz"]
            step["thought"] = "调用工具处理数据。"
            step["content"] = "调用工具处理数据。"
            step["dependencies"] = None
        degraded["thought"] = "调用工具完成任务。"
        reason = "工具全部错误且步骤内容极度模糊，双重缺陷"

    degraded["_degrade_reason"] = reason
    return degraded, reason


# ===================== 评分函数 =====================
def calculate_dependency_score(plan_json: dict) -> float:
    """
    依赖关系合理性：
    - 只有1步 → 无需依赖，给满分
    - 多步骤中至少有1步设置了依赖 → 说明规划者考虑了步骤间关系，给满分
    - 多步骤全部 dependencies=null → 可能遗漏了依赖关系，扣分
    """
    steps = plan_json.get("steps", [])
    if len(steps) <= 1:
        return 1.0
    has_any_dep = any(
        s.get("dependencies") and len(s["dependencies"]) > 0
        for s in steps
    )
    return 1.0 if has_any_dep else 0.5


def calculate_tool_score(used_tools: list, target_tools: list) -> float:
    """
    工具调用合理性：target_tools 是候选池，不要求全部调用。
    - 调用的工具全部在候选池内 → 1.0
    - 有不在候选池的幻觉/冗余工具 → 按比例扣分，最高0.7
    - 没有调用任何工具 → 0.0
    """
    used_set = set(used_tools)
    target_set = set(target_tools)
    if not target_set:
        return 1.0
    if not used_set:
        return 0.0
    valid_used = used_set & target_set
    invalid_used = used_set - target_set
    if invalid_used:
        precision = len(valid_used) / len(used_set)
        return round(precision * 0.7, 2)
    return 1.0


def calculate_step_score(step_count: int) -> float:
    if step_count == 3:
        return 1.0
    elif step_count in [2, 4]:
        return 0.9
    elif step_count == 1:
        return 0.4
    else:
        return 0.0


def calculate_content_score(plan_content: str, question: str) -> float:
    specific_patterns = re.findall(r'\d+|少于|大于|创建|筛选|返回|生成|提取|列出|获取', question.lower())
    if not specific_patterns:
        return 1.0
    if any(pattern in plan_content for pattern in specific_patterns):
        return 1.0
    elif any(kw in plan_content for kw in ["群体", "社交", "用户", "成员", "资料", "信息"]):
        return 0.5
    return 0.0


def calculate_keyword_score(plan_content: str, question: str, tool_keywords: list) -> float:
    stop_words = {"请", "调用", "和", "工具", "查询", "我", "在", "中", "当前", "关联", "的", "所有", "并", "为", "其中", "该", "此", "及", "返回", "列出"}
    question_cut = [kw.strip() for kw in re.split(r'[，。、()（）\s+]', question.lower()) if kw.strip()]
    business_keywords = [kw for kw in question_cut if kw not in stop_words and len(kw) > 2]
    core_kw = list(set(business_keywords + tool_keywords))
    if not core_kw:
        return 1.0
    matched_kw = [kw for kw in core_kw if kw in plan_content]
    matched_rate = len(matched_kw) / len(core_kw)
    if matched_rate >= 0.3 or len(matched_kw) >= 1:
        return 1.0
    return 0.5


def rule_score(plan_text: str, question: str, tools: list) -> float:
    plan_json = validate_json(plan_text)
    plan_content = ""
    if plan_json and "steps" in plan_json:
        for step in plan_json["steps"]:
            plan_content += step.get("title", "") + " " + step.get("content", "")
            # 同时纳入步骤级thought，增强内容评分覆盖
            plan_content += " " + step.get("thought", "")
    plan_content = plan_content.lower().replace('"', '').replace('[', '').replace(']', '')
    # tools字段现在可能是null，需要过滤
    used_tools = []
    if plan_json:
        for s in plan_json.get("steps", []):
            t = s.get("tools")
            if t and isinstance(t, list):
                used_tools.extend([x.lower().strip() for x in t])
    step_count = len(plan_json.get("steps", [])) if plan_json else 0
    tool_keywords = [t.lower().strip() for t in tools]
    # 额外维度：dependencies合理性（有依赖的步骤比例）
    has_dep_score = calculate_dependency_score(plan_json) if plan_json else 0.0

    dimensions = {
        "工具调用合理性": (0.35, lambda: calculate_tool_score(used_tools, tool_keywords), "调用工具均在候选池内，无幻觉工具"),
        "步骤逻辑性":     (0.25, lambda: calculate_step_score(step_count),                "步骤数2-4步，3步最优"),
        "内容具体性":     (0.20, lambda: calculate_content_score(plan_content, question),  "包含具体指标/操作，无模糊描述"),
        "依赖关系合理性": (0.10, lambda: has_dep_score,                                    "步骤间依赖关系是否合理设置"),
        "关键词匹配度":   (0.10, lambda: calculate_keyword_score(plan_content, question, tool_keywords), "匹配核心业务/工具关键词"),
    }

    total_score = 0.0
    print("  📊 维度评分明细：")
    for dim_name, (weight, score_func, desc) in dimensions.items():
        dim_score = score_func()
        total_score += dim_score * weight
        print(f"    - {dim_name}（权重{weight}）：{dim_score:.2f}分 → 加权{dim_score * weight:.2f}（{desc}）")

    final_score = round(1 + total_score * 4, 2)
    print(f"  🧮 Plan最终评分（1-5分制）：{final_score}")
    return final_score


# ===================== LLM对比评审（含位置偏差修正） =====================
def judge(question: str, plan_a: str, plan_b: str) -> str:
    """随机交换A/B位置，消除position bias"""
    swap = random.random() < 0.5
    real_first  = plan_b if swap else plan_a
    real_second = plan_a if swap else plan_b

    prompt = f"""
你是Ocoya平台的任务规划评审专家，需要严格按照以下标准判断哪个Plan更优。

### 评审标准（优先级从高到低）
1. 工具调用合理性：是否使用合适的工具解决问题，无冗余/遗漏
2. 步骤完整性：是否覆盖解决问题的核心环节
3. 步骤逻辑性：步骤顺序是否合理
4. 内容具体性：步骤内容是否具体可执行，避免模糊描述

### 用户问题
{question}

### Plan A
{real_first}

### Plan B
{real_second}

### 输出要求
仅输出JSON，无其他内容：{{"winner": "A" 或 "B", "reason": "简短评审理由"}}
若两个Plan质量相当，优先选择A。
"""
    try:
        judge_result = call_llm(prompt, temperature=config.temperature_judge)
        judge_json = validate_json(judge_result)
        if judge_json and judge_json.get("winner") in ["A", "B"]:
            raw_winner = judge_json["winner"]
            # 还原真实winner（消除swap影响）
            if swap:
                actual_winner = "B" if raw_winner == "A" else "A"
            else:
                actual_winner = raw_winner
            print(f"  🧑‍⚖️  评审结果：{actual_winner}（原始={raw_winner}, swap={swap}）| 理由：{judge_json.get('reason', '无')[:50]}...")
            return actual_winner
        else:
            raise ValueError("评审结果格式无效")
    except Exception as e:
        print(f"❌ 评审失败，随机选择：{e}")
        return random.choice(["A", "B"])


# ===================== 主流程 =====================
def main():
    preference_data, start_task_idx = load_resume_data()

    try:
        with open(config.input_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        if not isinstance(tasks, list):
            raise ValueError(f"输入文件必须是JSON列表格式")
        print(f"✅ 加载任务配置：共 {len(tasks)} 个任务")
        print(f"📌 负样本模式：{'主动降级构造' if config.use_degrade else '原始LLM对比'}")
    except Exception as e:
        print(f"❌ 加载任务配置失败：{e}")
        return

    # 多样性提示池
    diversity_hints = [
        "注意：请在方案中包含数据验证或结果确认步骤。",
        "注意：请考虑步骤间的数据依赖关系，明确说明上下步骤的数据传递。",
        "注意：请在关键步骤中说明若工具调用失败时的处理方式。",
        "注意：请尽量精简步骤，以最少步骤达成目标。",
        "注意：请在每个步骤中明确说明该步骤的输出数据格式。",
        "",  # 无额外提示，保持原始生成
        "",
        "",
    ]

    for task_idx in range(start_task_idx, len(tasks)):
        task = tasks[task_idx]
        if not all(k in task for k in ["theme", "tools"]):
            print(f"⚠️  跳过无效任务 {task_idx + 1}：缺少theme/tools字段")
            save_resume_data(preference_data, task_idx + 1)
            continue

        theme = task["theme"]
        raw_tools = task["tools"]
        print(f"\n========== 任务 {task_idx + 1}/{len(tasks)}: {theme} ==========")

        # 过滤工具：只保留与主题相关的工具
        tools = filter_tools_by_theme(theme, raw_tools)

        # 生成问题
        question = None
        for _ in range(config.max_retries):
            question = generate_question(theme, tools)
            if question and len(question) > 10:
                print(f"📝 生成用户问题：{question}")
                break
            print(f"⚠️  问题生成失败，重试...")
            time.sleep(config.api_delay)
        if not question:
            print(f"❌ 任务 {task_idx + 1} 问题生成失败，跳过")
            save_resume_data(preference_data, task_idx + 1)
            continue

        # 生成多个Plan
        valid_plans = []
        valid_scores = []
        max_attempts = config.num_plans * 3
        attempts = 0
        hint_pool = diversity_hints.copy()
        random.shuffle(hint_pool)

        while len(valid_plans) < config.num_plans and attempts < max_attempts:
            attempts += 1
            hint = hint_pool[len(valid_plans) % len(hint_pool)]
            print(f"  📋 生成Plan {len(valid_plans) + 1}/{config.num_plans}")
            plan = generate_plan(question, tools, diversity_hint=hint)
            if plan and validate_json(plan):
                score = rule_score(plan, question, tools)
                valid_plans.append(plan)
                valid_scores.append(score)
            else:
                print(f"  ❌ Plan无效，重新生成...（尝试 {attempts}/{max_attempts}）")
            time.sleep(config.api_delay)

        if len(valid_plans) < 2:
            print(f"⚠️  有效Plan数量不足（{len(valid_plans)}），跳过任务 {task_idx + 1}")
            save_resume_data(preference_data, task_idx + 1)
            continue

        # 取得分最高的Plan作为chosen基础
        ranked = sorted(zip(valid_plans, valid_scores), key=lambda x: x[1], reverse=True)
        top_plans = [x[0] for x in ranked[:config.top_k]]
        discard_count = 0

        # ===================== 构造偏好对 =====================
        for plan_a_text in top_plans:
            plan_a_json = validate_json(plan_a_text)

            if config.use_degrade:
                # 模式1：主动降级构造负样本（推荐）
                degraded_json, degrade_reason = degrade_plan(plan_a_json, tools)
                # 移除内部调试字段，避免污染训练数据
                degraded_json.pop("_degrade_reason", None)

                # 仍用judge做一轮验证（确认chosen确实优于rejected）
                plan_b_text = json.dumps(degraded_json, ensure_ascii=False)
                winner = judge(question, plan_a_text, plan_b_text)

                if winner == "A":
                    preference_data.append({
                        "prompt":           question,
                        "chosen":           plan_a_json,
                        "rejected":         degraded_json,
                        "degrade_strategy": degrade_reason,
                        "task_theme":       theme,
                        "generate_time":    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    })
                else:
                    # 降级版意外胜出 → 说明该降级策略对此Plan无效，直接丢弃这对数据
                    # 绝不能把降级版当chosen存入，否则污染训练数据
                    print(f"  ⚠️  降级Plan意外胜出，丢弃此对（strategy={degrade_reason}）")
                    discard_count += 1

            else:
                # 模式2：原始LLM对比（保留兼容）
                bottom_plans = [x[0] for x in ranked[-config.top_k:]]
                for plan_b_text in bottom_plans:
                    if plan_a_text == plan_b_text:
                        continue
                    winner = judge(question, plan_a_text, plan_b_text)
                    chosen_text  = plan_a_text if winner == "A" else plan_b_text
                    rejected_text = plan_b_text if winner == "A" else plan_a_text
                    preference_data.append({
                        "prompt":        question,
                        "chosen":        validate_json(chosen_text),
                        "rejected":      validate_json(rejected_text),
                        "task_theme":    theme,
                        "generate_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    })
                    time.sleep(config.api_delay)

            time.sleep(config.api_delay)

        save_resume_data(preference_data, task_idx + 1)
        print(f"✅ 任务 {task_idx + 1} 完成，累计生成 {len(preference_data)} 条样本"
              + (f"（丢弃 {discard_count} 对无效降级）" if discard_count else ""))

    # 保存最终结果
    try:
        with open(config.output_file, "w", encoding="utf-8") as f:
            json.dump(preference_data, f, ensure_ascii=False, indent=2)
        print(f"\n🎉 所有任务完成！")
        print(f"📊 总样本数：{len(preference_data)}")
        print(f"💾 保存路径：{os.path.abspath(config.output_file)}")
        if os.path.exists(config.resume_file):
            os.remove(config.resume_file)
            print(f"🧹 清理断点文件：{config.resume_file}")
    except Exception as e:
        print(f"❌ 保存结果失败：{e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 程序执行失败：{e}")
        traceback.print_exc()

import re
import json
import numpy as np
from collections import defaultdict, Counter
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import hdbscan

# ============================================================
# 0. 配置
# ============================================================

MIN_TOOLS_PER_SCENE = 2
MIN_TOOLS_PER_SERVICE = 3
MIN_SERVICE_SUPPORT = 5

# HDBSCAN 参数
HDBSCAN_MIN_CLUSTER_SIZE = 2   # 簇最少工具数
HDBSCAN_MIN_SAMPLES = 1        # 越小越宽松，噪声点越少
# 编码模型
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

STOP_TOOLS = {
    "Finish", "give_up_and_restart", "confirm_for_auth",
    "getuser_for_auth", "getuserlist_for_auth"
}
TOOL_PATTERN = re.compile(r"^[a-z][a-z0-9_]+_for_[a-z0-9_]+$", re.IGNORECASE)


# ============================================================
# 1. 工具解析
# ============================================================

def valid_tool(tool: str) -> bool:
    return tool not in STOP_TOOLS and TOOL_PATTERN.match(tool) is not None


def tool_service(tool: str) -> str:
    return tool.split("_for_")[-1]


def tool_to_text(tool: str) -> str:
    """把工具名转成可读文本用于编码：下划线替换成空格"""
    return tool.replace("_", " ")


def extract_tool_chain(sample):
    tools = []
    for turn in sample.get("conversations", []):
        if turn.get("from") == "assistant":
            m = re.search(r"Action:\s*([a-zA-Z0-9_]+)", turn.get("value", ""))
            if m:
                t = m.group(1)
                if valid_tool(t):
                    tools.append(t)
    return list(dict.fromkeys(tools))


def load_simple_scenes():
    dataset = load_dataset("Yhyu13/ToolBench_toolllama_G123_dfs", split="train")
    scenes = []
    for sample in dataset:
        tools = extract_tool_chain(sample)
        if len(tools) >= MIN_TOOLS_PER_SCENE:
            scenes.append(tools)
    print(f"✓ Parsed {len(scenes)} scenes")
    return scenes


# ============================================================
# 2. Service 聚合
# ============================================================

def build_service_profiles(scenes):
    service_data = defaultdict(lambda: {"tools": set(), "scene_count": 0, "scenes_using": []})
    for idx, scene in enumerate(scenes):
        seen_services = defaultdict(list)
        for tool in scene:
            seen_services[tool_service(tool)].append(tool)
        for service, tools in seen_services.items():
            service_data[service]["tools"].update(tools)
            service_data[service]["scene_count"] += 1
            service_data[service]["scenes_using"].append(idx)

    filtered = {
        s: d for s, d in service_data.items()
        if len(d["tools"]) >= MIN_TOOLS_PER_SERVICE
        and d["scene_count"] >= MIN_SERVICE_SUPPORT
    }
    print(f"✓ {len(filtered)} services after filtering (dropped {len(service_data)-len(filtered)})")
    return filtered


# ============================================================
# 3. 语义编码
# ============================================================

def embed_tools(all_tools: list, model) -> np.ndarray:
    """把工具名列表编码成矩阵，行对应工具"""
    texts = [tool_to_text(t) for t in all_tools]
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    return embeddings


# ============================================================
# 4. HDBSCAN 语义聚类
# ============================================================

def cluster_tools_semantic(tools: list, embeddings: np.ndarray) -> dict[str, list]:
    """
    对一个 service 内的工具做 HDBSCAN 语义聚类。
    大簇（>15个工具）自动提高 min_cluster_size，强制切得更细。
    噪声点（label=-1）如果数量够多则单独成簇，否则丢弃。
    """
    if len(tools) <= HDBSCAN_MIN_CLUSTER_SIZE:
        return {"0": tools}

    # 大簇动态调整 min_cluster_size，让 HDBSCAN 切得更细
    n = len(tools)
    if n > 15:
        adaptive_min_cluster = max(2, n // 5)   # 每簇约占总数的20%
    else:
        adaptive_min_cluster = HDBSCAN_MIN_CLUSTER_SIZE

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=adaptive_min_cluster,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="leaf",  # leaf 比 eom 切得更细
    )
    labels = clusterer.fit_predict(embeddings)

    groups = defaultdict(list)
    for tool, label in zip(tools, labels):
        groups[str(label)].append(tool)

    # 噪声点单独处理：够多就保留，否则丢弃
    noise = groups.pop("-1", [])
    if len(noise) >= 2:
        groups["noise"] = noise

    # 如果全部是噪声（聚类完全失败），退回整体作为一个簇
    if not groups:
        return {"0": tools}

    return dict(groups)


# ============================================================
# 5. Fallback：HDBSCAN 未切分时按功能前缀做关键词切分
# ============================================================

FALLBACK_FUNC_GROUPS = [
    # 通用操作
    ["search", "find", "lookup", "query"],
    ["by_class", "by_type", "by_race", "by_faction", "by_quality", "by_set", "by_category"],
    ["get", "single", "detail", "info", "about"],
    ["all", "list", "index", "catalog"],
    ["create", "add", "post", "insert"],
    ["update", "edit", "modify"],
    ["delete", "remove"],
    ["forecast", "predict", "future"],
    ["current", "live", "realtime", "now"],
    ["history", "historical", "past", "stats"],
    ["top", "trending", "popular", "rank", "best"],
    # 金融技术指标（移动均线类）
    ["ema", "sma", "wma", "dema", "tema", "trima", "t3ma", "mama", "ma_"],
    # 金融技术指标（动量/震荡类）
    ["rsi", "cci", "mom", "roc", "rocr", "cmo", "crsi", "mfi", "willr", "ultosc", "coppock"],
    # 金融技术指标（趋势类）
    ["adx", "aroon", "sar", "apo", "ppo", "macd", "dmi", "adosc", "obv"],
    # 金融技术指标（统计/数学类）
    ["stddev", "var", "beta", "correl", "linearreg", "sqrt", "ln", "ceil", "floor",
     "min_", "max_", "avg", "sum", "sub", "div", "midpoint", "midprice", "medprice",
     "avgprice", "wclprice", "minmax", "minus_di", "percent_b", "ht_"],
    # 金融基本面
    ["earnings", "balance_sheet", "dividends", "eps", "growth", "institutional",
     "sustainability", "recommendations", "analyst", "profile", "statistics", "risk"],
    # 金融市场数据
    ["market_movers", "quote", "real_time_price", "time_series", "options",
     "ipo", "composition", "exchanges", "crypto_exchanges", "currency_conversion",
     "earliest_timestamp", "symbol_search", "logo"],
    # 体育赛事类
    ["match", "fixture", "result", "standing", "score", "league", "season"],
    ["player", "team", "coach", "squad", "transfer"],
    ["live", "odds", "lineup", "event"],
    # 用户名/账号检查类
    ["instagram", "twitter", "facebook", "tiktok", "snapchat", "reddit",
     "youtube", "twitch", "github", "pinterest", "tumblr", "telegram"],
]

def fallback_keyword_split(tools: list) -> dict[str, list]:
    """
    HDBSCAN 未能切分时的兜底方案：按工具名功能关键词分组。
    匹配不到任何组的工具归入 misc。
    """
    groups = defaultdict(list)
    for tool in tools:
        tool_lower = tool.lower()
        matched = False
        for group_kws in FALLBACK_FUNC_GROUPS:
            if any(kw in tool_lower for kw in group_kws):
                groups[group_kws[0]].append(tool)  # 用第一个关键词作为组名
                matched = True
                break
        if not matched:
            groups["misc"].append(tool)

    # 过滤太小的组（合并到 misc）
    result = {}
    misc = list(groups.get("misc", []))
    for label, group_tools in groups.items():
        if label == "misc":
            continue
        if len(group_tools) < 2:
            misc.extend(group_tools)
        else:
            result[label] = group_tools
    if misc:
        result["misc"] = misc

    return result if len(result) > 1 else {"0": tools}  # 切不开就原样返回


# ============================================================
# 6. 生成 theme 描述（用 service 名 + 簇内高频词）
# ============================================================

def infer_cluster_theme(service: str, tools: list) -> str:
    """从 service 名和工具名提取语义化主题描述"""
    # 提取工具名功能部分（_for_ 前的词）的高频词
    word_counter = Counter()
    for tool in tools:
        func_part = tool.split("_for_")[0]
        words = func_part.split("_")
        for w in words:
            if len(w) > 2:  # 过滤太短的词
                word_counter[w] += 1

    # 取前2个高频词作为功能描述
    top_words = [w for w, _ in word_counter.most_common(2)]
    service_readable = service.replace("_", " ")
    func_desc = " ".join(top_words) if top_words else "general"

    return f"{func_desc} {service_readable}".strip()


# ============================================================
# 7. 主流程：构建细粒度 compounds
# ============================================================

def build_semantic_compounds(scenes, service_profiles, model):
    # 收集所有工具 + 对应 service
    all_tools_ordered = []
    tool_to_service = {}
    for service, profile in service_profiles.items():
        for tool in profile["tools"]:
            if tool not in tool_to_service:
                all_tools_ordered.append(tool)
                tool_to_service[tool] = service

    print(f"\n🔢 Encoding {len(all_tools_ordered)} tools...")
    all_embeddings = embed_tools(all_tools_ordered, model)
    tool_to_embedding = {t: all_embeddings[i] for i, t in enumerate(all_tools_ordered)}

    compounds = []
    compound_id = 0

    for service, profile in service_profiles.items():
        service_tools = sorted(profile["tools"])
        scene_indices = set(profile["scenes_using"])

        if len(service_tools) < 2:
            continue

        # 取该 service 工具的 embedding 子矩阵
        service_embeddings = np.stack([tool_to_embedding[t] for t in service_tools])

        # HDBSCAN 聚类
        clusters = cluster_tools_semantic(service_tools, service_embeddings)

        # 如果整个 service 只产出1个簇且工具数>6，fallback 到关键词切分
        if len(clusters) == 1 and len(service_tools) > 6:
            only_label = list(clusters.keys())[0]
            if only_label != "-1":
                fallback = fallback_keyword_split(service_tools)
                if len(fallback) > 1:
                    clusters = fallback

        # 对仍然超过20个工具的簇，强制再做一次fallback关键词切分
        expanded_clusters = {}
        for label, cluster_tools in clusters.items():
            if len(cluster_tools) > 20:
                sub = fallback_keyword_split(cluster_tools)
                if len(sub) > 1:
                    for sub_label, sub_tools in sub.items():
                        expanded_clusters[f"{label}_{sub_label}"] = sub_tools
                else:
                    expanded_clusters[label] = cluster_tools
            else:
                expanded_clusters[label] = cluster_tools
        clusters = expanded_clusters

        for cluster_label, cluster_tools in clusters.items():
            # 噪声点或过小的簇直接跳过
            if len(cluster_tools) < 2:
                continue
            # 超过20个工具的簇标记为large，保留进test_large.json
            if len(cluster_tools) > 20:
                print(f"  📦 超大簇 {service}/{cluster_label}：{len(cluster_tools)} 个工具 → test_large.json")

            # 计算 support
            chunk_set = set(cluster_tools)
            support = sum(
                1 for idx in scene_indices
                if any(t in chunk_set for t in scenes[idx])
            )

            theme_desc = infer_cluster_theme(service, cluster_tools)
            compound_id += 1

            compounds.append({
                "compound_id": f"compound_{compound_id:05d}",
                "theme": theme_desc,
                "service": service,
                "cluster_label": cluster_label,
                "tools": sorted(cluster_tools),
                "num_tools": len(cluster_tools),
                "num_simple_scenes": support,
            })

    # 按 support 降序排列
    compounds.sort(key=lambda x: x["num_simple_scenes"], reverse=True)
    return compounds


# ============================================================
# 7. 转换为 test.json
# ============================================================

def convert_to_test_json(compounds: list) -> list:
    return [
        {
            "theme": c["theme"],
            "tools": c["tools"],
            "num_tools": c["num_tools"],
        }
        for c in compounds
    ]


# ============================================================
# 8. 主入口
# ============================================================

def main():
    print("=" * 60)
    print("语义聚类：sentence-transformers + HDBSCAN")
    print("=" * 60)

    scenes = load_simple_scenes()

    print("\n[Stage 1] Building service profiles...")
    service_profiles = build_service_profiles(scenes)

    print("\n[Stage 2] Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    print("\n[Stage 3] Semantic clustering...")
    compounds = build_semantic_compounds(scenes, service_profiles, model)

    print(f"\n✓ Generated {len(compounds)} compounds")

    # 分布统计
    tool_dist = Counter(c["num_tools"] for c in compounds)
    print("\n📊 Tools-per-compound distribution:")
    for n in sorted(tool_dist):
        print(f"  {n} tools: {tool_dist[n]} compounds")

    # 保存
    with open("toolbench_semantic_compounds.json", "w", encoding="utf-8") as f:
        json.dump(compounds, f, ensure_ascii=False, indent=2)
    print("\n✓ Saved to toolbench_semantic_compounds.json")

    tasks = convert_to_test_json(compounds)
    tasks_small = [t for t in tasks if len(t["tools"]) <= 10]
    tasks_large = [t for t in tasks if len(t["tools"]) > 10]

    with open("test_small.json", "w", encoding="utf-8") as f:
        json.dump(tasks_small, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved {len(tasks_small)} tasks (≤10 tools) to test_small.json")

    with open("test_large.json", "w", encoding="utf-8") as f:
        json.dump(tasks_large, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved {len(tasks_large)} tasks (>10 tools) to test_large.json")

    # 样例
    with open("sample_semantic.json", "w", encoding="utf-8") as f:
        json.dump(compounds[:5], f, ensure_ascii=False, indent=2)
    print("✓ Saved sample to sample_semantic.json")


if __name__ == "__main__":
    main()

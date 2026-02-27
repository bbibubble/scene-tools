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
# 编码模型（轻量，本地运行快）
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
# 4. 【核心】HDBSCAN 语义聚类
# ============================================================

def cluster_tools_semantic(tools: list, embeddings: np.ndarray) -> dict[str, list]:
    """
    对一个 service 内的工具做 HDBSCAN 语义聚类。
    返回 {cluster_id_str: [tool, ...]}
    噪声点（label=-1）单独归为 'noise' 组后按 MAX 切分。
    """
    if len(tools) <= HDBSCAN_MIN_CLUSTER_SIZE:
        return {"0": tools}

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",   # eom 比 leaf 更合并
    )
    labels = clusterer.fit_predict(embeddings)

    groups = defaultdict(list)
    for tool, label in zip(tools, labels):
        groups[str(label)].append(tool)

    return dict(groups)


# ============================================================
# 5. 生成 theme 描述（用 service 名 + 簇内高频词）
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
# 6. 主流程：构建细粒度 compounds
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

        for cluster_label, cluster_tools in clusters.items():
            # 噪声点或过小的簇直接跳过
            if len(cluster_tools) < 2:
                continue

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
            "_compound_id": c["compound_id"],
            "_service": c["service"],
            "_num_scenes": c["num_simple_scenes"],
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

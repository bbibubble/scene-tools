# ============================================================
# ToolBench → Tool Sets via Graph Community Detection
# ============================================================

import os
import re
import json
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import pickle

from datasets import load_dataset
import networkx as nx
from networkx.algorithms import community
import multiprocessing as mp

# ------------------------------------------------------------
# 0. 全局参数
# ------------------------------------------------------------

CACHE_PARSED = "parsed_data_graph.pkl"

MAX_SCENE_LEN = 256
BATCH_SIZE = 256

# **图方法参数**
MIN_COOCCURRENCE = 3  # 工具共现最小次数
EDGE_WEIGHT_THRESHOLD = 0.1  # 边权重阈值
RESOLUTION = 1.0  # Louvain分辨率（越大越多社区）

CORE_THRESHOLD = 0.2
MIN_TOOLS_PER_SET = 2
MAX_TOOLS_PER_SET = 15
MIN_TOOLSET_SUPPORT = 5

INVALID_TOOLS = {"Finish", "give_up_and_restart"}

TOOL_NAME_PATTERN = re.compile(
    r'^[a-z][a-z0-9_]{3,}_for_[a-z][a-z0-9_]{2,}$',
    re.IGNORECASE
)


# ------------------------------------------------------------
# 1. 数据解析（保持不变）
# ------------------------------------------------------------
def is_valid_tool(tool):
    if tool in INVALID_TOOLS:
        return False
    if not TOOL_NAME_PATTERN.match(tool):
        return False
    parts = tool.split('_for_')
    if len(parts) != 2:
        return False
    action, service = parts
    if len(action) < 4 or len(service) < 3:
        return False
    return True


def normalize_scene(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()[:MAX_SCENE_LEN]


def extract_scene_and_chain(sample):
    conversations = sample.get("conversations", [])
    scene = None
    tool_chain = []

    for turn in conversations:
        if turn.get("from") == "user" and scene is None:
            scene = normalize_scene(turn.get("value", ""))

        if turn.get("from") == "assistant":
            m = re.search(r"Action:\s*([a-zA-Z0-9_]+)", turn.get("value", ""))
            if m:
                tool = m.group(1)
                if is_valid_tool(tool):
                    tool_chain.append(tool)

    if scene and len(tool_chain) >= MIN_TOOLS_PER_SET:
        return scene, tool_chain
    return None


def parse_dataset_parallel(data, num_workers=4):
    if os.path.exists(CACHE_PARSED):
        print("Loading cached parsed data...")
        with open(CACHE_PARSED, 'rb') as f:
            return pickle.load(f)

    print(f"Parsing dataset with {num_workers} workers...")
    data_list = list(data)

    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(extract_scene_and_chain, data_list, chunksize=100),
            total=len(data_list),
            desc="Parsing"
        ))

    scene_chains = defaultdict(list)
    for res in results:
        if res:
            scene, chain = res
            scene_chains[scene].append(chain)

    with open(CACHE_PARSED, 'wb') as f:
        pickle.dump(scene_chains, f)

    return scene_chains


def load_toolbench():
    return load_dataset(
        "Yhyu13/ToolBench_toolllama_G123_dfs",
        split="train"
    )


# ------------------------------------------------------------
# 2. **核心：构建工具共现图**
# ------------------------------------------------------------
def build_tool_cooccurrence_graph(scene_chains):
    """
    构建加权无向图：
    - 节点 = 工具
    - 边 = 工具在同一个tool chain中共现
    - 权重 = 共现次数 / PMI得分
    """
    print("Building tool co-occurrence graph...")

    # 统计工具出现次数
    tool_counts = Counter()
    pair_counts = defaultdict(int)
    total_chains = 0

    for scene, chains in tqdm(scene_chains.items(), desc="Counting"):
        for chain in chains:
            total_chains += 1
            unique_tools = list(set(chain))  # 去重

            # 统计单个工具
            for tool in unique_tools:
                tool_counts[tool] += 1

            # 统计工具对
            for i, tool1 in enumerate(unique_tools):
                for tool2 in unique_tools[i + 1:]:
                    pair = tuple(sorted([tool1, tool2]))
                    pair_counts[pair] += 1

    # 构建图
    G = nx.Graph()

    # 添加节点
    for tool, count in tool_counts.items():
        G.add_node(tool, frequency=count)

    # 添加边（使用PMI作为权重）
    print("Computing edge weights (PMI)...")
    for (tool1, tool2), co_count in tqdm(pair_counts.items(), desc="Edges"):
        if co_count < MIN_COOCCURRENCE:
            continue

        # PMI (Pointwise Mutual Information)
        p_tool1 = tool_counts[tool1] / total_chains
        p_tool2 = tool_counts[tool2] / total_chains
        p_both = co_count / total_chains

        pmi = np.log2(p_both / (p_tool1 * p_tool2))

        # 归一化到 [0, 1]
        normalized_weight = max(0, pmi / 10)  # PMI通常在[-10, 10]

        if normalized_weight > EDGE_WEIGHT_THRESHOLD:
            G.add_edge(tool1, tool2, weight=normalized_weight, count=co_count)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G, tool_counts


# ------------------------------------------------------------
# 3. **社区发现算法**
# ------------------------------------------------------------
def detect_tool_communities(G, method='louvain'):
    """
    使用社区发现算法分组工具

    方法选择：
    - louvain: 快速，适合大图（推荐）
    - girvan_newman: 层次化，可解释性强
    - label_propagation: 极快，但不稳定
    """
    print(f"Detecting communities using {method}...")

    if method == 'louvain':
        # Louvain算法（最常用）
        communities = community.louvain_communities(
            G,
            weight='weight',
            resolution=RESOLUTION,
            seed=42
        )

    elif method == 'greedy_modularity':
        # 贪心模块度优化
        communities = community.greedy_modularity_communities(
            G,
            weight='weight'
        )

    elif method == 'label_propagation':
        # 标签传播（最快）
        communities = community.label_propagation_communities(G)

    else:
        raise ValueError(f"Unknown method: {method}")

    # 转换为字典格式
    tool_to_community = {}
    for comm_id, comm in enumerate(communities):
        for tool in comm:
            tool_to_community[tool] = comm_id

    print(f"Found {len(communities)} communities")

    return communities, tool_to_community


# ------------------------------------------------------------
# 4. **从社区构建工具集**
# ------------------------------------------------------------
def build_toolsets_from_communities(communities, scene_chains, tool_counts):
    """
    将每个社区转换为一个工具集
    """
    print("Building toolsets from communities...")

    toolsets = []

    for comm_id, community_tools in enumerate(tqdm(communities, desc="Communities")):
        community_tools = list(community_tools)

        # 过滤大小
        if not (MIN_TOOLS_PER_SET <= len(community_tools) <= MAX_TOOLS_PER_SET):
            continue

        # 收集使用该社区工具的场景
        matching_scenes = []
        all_chains = []
        tool_usage = Counter()

        for scene, chains in scene_chains.items():
            for chain in chains:
                chain_tools = set(chain)
                community_set = set(community_tools)

                # 如果chain中至少有2个工具来自该社区
                overlap = len(chain_tools & community_set)
                if overlap >= 2:
                    matching_scenes.append(scene)
                    all_chains.append(chain)
                    tool_usage.update(chain_tools & community_set)

        # 去重场景
        matching_scenes = list(set(matching_scenes))

        if len(matching_scenes) < MIN_TOOLSET_SUPPORT:
            continue

        # 计算 core/optional
        total = sum(tool_usage.values())
        core = [t for t in community_tools if tool_usage[t] / total >= CORE_THRESHOLD]
        optional = [t for t in community_tools if t not in core]

        # 如果没有core工具，强制选择最频繁的
        if not core:
            top_tools = [t for t, _ in tool_usage.most_common(min(3, len(community_tools)))]
            core = top_tools
            optional = [t for t in community_tools if t not in core]

        toolsets.append({
            "community_id": comm_id,
            "support": len(matching_scenes),
            "scene_count": len(matching_scenes),
            "example_scenes": matching_scenes[:5],
            "core_tools": sorted(core),
            "optional_tools": sorted(optional),
            "tool_chains": all_chains[:5],
            "tool_usage": dict(tool_usage.most_common()),
            "avg_tool_frequency": float(np.mean([tool_counts[t] for t in community_tools]))
        })

    return toolsets


# ------------------------------------------------------------
# 5. 主流程
# ------------------------------------------------------------
def build_tool_sets_graph_based(num_workers=4, method='louvain'):
    """
    基于图的工具集构建
    """
    print("=" * 60)
    print("Tool Set Builder - Graph-based Approach")
    print("=" * 60)

    # 加载数据
    print("\n[1/5] Loading data...")
    data = load_toolbench()
    scene_chains = parse_dataset_parallel(data, num_workers=num_workers)
    print(f"  Valid scenes: {len(scene_chains)}")

    # 构建图
    print("\n[2/5] Building co-occurrence graph...")
    G, tool_counts = build_tool_cooccurrence_graph(scene_chains)

    # 社区发现
    print("\n[3/5] Detecting tool communities...")
    communities, tool_to_comm = detect_tool_communities(G, method=method)

    # 构建工具集
    print("\n[4/5] Building toolsets...")
    toolsets = build_toolsets_from_communities(communities, scene_chains, tool_counts)

    # 排序和ID
    toolsets.sort(key=lambda x: x['support'], reverse=True)
    for i, ts in enumerate(toolsets):
        ts['toolset_id'] = i

    print("\n[5/5] Saving results...")

    return toolsets, G


# ------------------------------------------------------------
# 6. 执行 + 可视化
# ------------------------------------------------------------
if __name__ == "__main__":
    NUM_WORKERS = min(8, mp.cpu_count())

    toolsets, graph = build_tool_sets_graph_based(
        num_workers=NUM_WORKERS,
        method='louvain'  # 可选：'greedy_modularity', 'label_propagation'
    )

    # 保存结果
    with open("toolbench_toolsets_graph.json", "w", encoding="utf-8") as f:
        json.dump(toolsets, f, ensure_ascii=False, indent=2)

    # 统计
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    print(f"Total toolsets: {len(toolsets)}")
    print(f"With core tools: {sum(1 for ts in toolsets if ts['core_tools'])}")
    print(f"Avg core tools: {np.mean([len(ts['core_tools']) for ts in toolsets]):.1f}")
    print(f"Avg optional tools: {np.mean([len(ts['optional_tools']) for ts in toolsets]):.1f}")
    print(f"Avg support: {np.mean([ts['support'] for ts in toolsets]):.1f}")

    print(f"\n🔝 Top 10 Tool Sets:")
    for ts in toolsets[:10]:
        print(f"\n  #{ts['toolset_id']} (support={ts['support']}, scenes={ts['scene_count']})")
        print(f"    Core ({len(ts['core_tools'])}): {ts['core_tools'][:3]}")
        print(f"    Optional ({len(ts['optional_tools'])}): {ts['optional_tools'][:3]}")
        print(f"    Scene: {ts['example_scenes'][0][:80]}...")

    # 保存图结构（可选）
    print("\n💾 Saving graph...")
    nx.write_gexf(graph, "tool_cooccurrence_graph.gexf")
    print("  → Can be visualized in Gephi")

    # 简单可视化（如果安装了matplotlib）
    try:
        import matplotlib.pyplot as plt

        print("\n📈 Generating graph visualization...")

        # 只可视化最大连通分量
        largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc)

        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50)

        # 节点大小 = 工具使用频率
        node_sizes = [graph.nodes[n]['frequency'] * 10 for n in subgraph.nodes()]

        nx.draw_networkx(
            subgraph,
            pos,
            node_size=node_sizes,
            node_color='lightblue',
            font_size=6,
            with_labels=True,
            edge_color='gray',
            alpha=0.7
        )

        plt.title("Tool Co-occurrence Graph (Largest Component)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("tool_graph_visualization.png", dpi=150, bbox_inches='tight')
        print("  → Saved to tool_graph_visualization.png")

    except ImportError:
        print("  (Skipping visualization - install matplotlib to enable)")
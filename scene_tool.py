import re
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from datasets import concatenate_datasets


# -------------------------------------------------
# 1. 从 ToolBench 样本中提取 scene + tools
# -------------------------------------------------
def extract_scene_tools(sample):
    conversations = sample.get("conversations", [])

    scene = None
    tools = []

    for turn in conversations:
        # scene：优先 user
        if turn.get("from") == "user" and scene is None:
            scene = turn.get("value", "").strip()

        # tools：从 assistant 的 Action 中抽取
        if turn.get("from") == "assistant":
            text = turn.get("value", "")
            m = re.search(r"Action:\s*([a-zA-Z0-9_]+)", text)
            if m:
                tools.append(m.group(1))

    if not scene or not tools:
        return None

    return {
        "scene": scene,
        "tools": tools
    }


# -------------------------------------------------
# 2. 加载 ToolBench 数据集
# -------------------------------------------------
def load_toolbench():
  ds = load_dataset("Yhyu13/ToolBench_toolllama_G123_dfs")
  #ds = load_dataset("", "default")
  return concatenate_datasets([ds["train"], ds["test"]])


# -------------------------------------------------
# 3. 主流程：提取 → 聚类 → 工具集生成
# -------------------------------------------------
def build_tool_sets():
    print("Loading ToolBench dataset...")
    data = load_toolbench()

    print("Extracting scenes and tools...")
    pairs = []
    for sample in tqdm(data):
        item = extract_scene_tools(sample)
        if item:
            pairs.append(item)

    scenes = [p["scene"] for p in pairs]
    tools_list = [p["tools"] for p in pairs]

    print(f"Valid samples: {len(scenes)}")

    # -------------------------------------------------
    # 4. Scene Embedding
    # -------------------------------------------------
    print("Encoding scenes...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    scene_embs = model.encode(
        scenes,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # -------------------------------------------------
    # 5. 场景聚类
    # -------------------------------------------------
    print("Clustering scenes...")
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.35,
        metric="cosine",
        linkage="average"
    )

    labels = clusterer.fit_predict(scene_embs)

    # -------------------------------------------------
    # 6. 聚合工具
    # -------------------------------------------------
    cluster_tools = defaultdict(list)
    cluster_scenes = defaultdict(list)

    for i, label in enumerate(labels):
        cluster_scenes[label].append(scenes[i])
        cluster_tools[label].extend(tools_list[i])

    tool_sets = []
    for cid in cluster_tools:
        freq = defaultdict(int)
        for t in cluster_tools[cid]:
            freq[t] += 1

        tool_sets.append({
            "cluster_id": int(cid),
            "scene_count": len(cluster_scenes[cid]),
            "tools": sorted(freq.keys()),
            "tool_frequency": dict(freq),
            "example_scenes": cluster_scenes[cid][:3]
        })

    return tool_sets


# -------------------------------------------------
# 7. 执行 & 保存
# -------------------------------------------------
if __name__ == "__main__":
    tool_sets = build_tool_sets()

    with open("toolbench_clustered_tool_sets.json", "w", encoding="utf-8") as f:
        json.dump(tool_sets, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(tool_sets)} tool sets.")

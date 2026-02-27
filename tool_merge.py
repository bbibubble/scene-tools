import re
import json
from collections import defaultdict, Counter
from datasets import load_dataset

# ============================================================
# 0. 配置
# ============================================================

MIN_TOOLS_PER_SCENE = 2
MIN_TOOLS_PER_SERVICE = 3
MIN_SERVICE_SUPPORT = 5
MIN_THEME_PURITY = 0.55  # 主题纯度阈值

STOP_TOOLS = {
    "Finish",
    "give_up_and_restart",
    "confirm_for_auth",
    "getuser_for_auth",
    "getuserlist_for_auth"
}

TOOL_PATTERN = re.compile(
    r"^[a-z][a-z0-9_]+_for_[a-z0-9_]+$",
    re.IGNORECASE
)

# ============================================================
# 1. 主题分类
# ============================================================

THEME_KEYWORDS = {
    "geography": [
        "location", "place", "geo", "coordinate", "map", "nearby", "radius",
        "city", "cities", "country", "continent", "timezone", "address", "distance"
    ],

    "sports": [
        "sport", "match", "team", "player", "athlete", "coach", "score",
        "nba", "nfl", "nhl", "mlb", "soccer", "football", "basketball",
        "league", "tournament", "championship", "standings", "fixture",
        "fifa", "ufc", "fight", "boxing", "tennis", "golf", "racing",
        "transfer", "eredivisie", "formula", "cricket"
    ],

    "gaming": [
        "game", "gaming", "gamer", "esport", "steam", "epic", "playstation",
        "minecraft", "valorant", "csgo", "dota", "fortnite", "pokemon",
        "rpg", "mmo", "fps", "dungeon", "hero", "weapon", "champion",
        "skin", "avatar", "quest", "level", "inventory", "twitch", "lost_ark",
        "hearthstone", "rocket_league", "trackmania", "warzone"
    ],

    "puzzle_games": [
        "puzzle", "chess", "sudoku", "wordle", "crossword", "scrabble",
        "word", "charade", "pictionary", "trivia", "bingo", "dice", "card"
    ],

    "food": [
        "food", "recipe", "restaurant", "drink", "cocktail", "dessert",
        "nutrition", "meal", "cook", "ingredient", "dish", "burger",
        "pizza", "keto", "vegan", "diet", "beverage", "cheese"
    ],

    "social_media": [
        "user", "profile", "tweet", "media", "hashtag", "follower",
        "following", "post", "comment", "like", "share", "instagram",
        "tiktok", "facebook", "twitter", "social", "streaming", "track", "lyric", "concert", "band", "genre"
    ],

    "finance": [
        "account", "transaction", "payment", "invoice", "balance",
        "currency", "exchange", "forex", "stock", "crypto", "price",
        "rate", "conversion", "trading", "market", "invest", "wallet",
        "coin", "dividend", "earning",
        # 新增：
        "finance", "financial", "bank", "fund", "portfolio", "bond",
        "commodity", "futures", "option", "ticker", "quote", "equity",
        "economic", "cash", "debt", "asset", "revenue", "profit",
        "gas_price", "metal_price", "energy_price",  # 大宗商品价格
        "stocktwits", "investing", "trader", "capital"  # 投资社交
    ],

    "commerce": [
        "product", "cart", "order", "sale", "catalog", "inventory",
        "shop", "store", "purchase", "customer", "checkout", "shipping"
    ],

    "media": [
        "music", "album", "playlist", "video", "podcast", "audio",
        "song", "artist", "movie", "film", "spotify", "shazam",
        "billboard", "deezer", "soundcloud", "youtube"
    ],

    "education": [
        "study", "course", "lesson", "concept", "learning", "tutorial",
        "teach", "exam", "quiz", "test", "school", "kanji", "japanese"
    ],

    "lifestyle": [
        "weather", "forecast", "climate", "temperature", "rain", "wind",
        "event", "booking", "service", "schedule", "hotel", "travel",
        "air", "quality", "astronomy", "vacation", "flight", "airport", "rental", "tourism"
    ],

    "entertainment": [
        "joke", "meme", "humor", "funny", "comedy", "laugh", "prank",
        "quote", "superhero", "villain", "marvel", "comic", "anime"
    ],

    "translation": [
        "translate", "translation", "language", "dictionary", "synonym",
        "antonym", "definition", "text", "speech"
    ],

    "news": [
        "news", "article", "headline", "press", "journalist", "report",
        "story", "breaking", "update", "blog"
    ]
}

DEFAULT_THEME = "other"


# ============================================================
# 2. Tool 解析
# ============================================================

def valid_tool(tool: str) -> bool:
    if tool in STOP_TOOLS:
        return False
    return TOOL_PATTERN.match(tool) is not None


def tool_service(tool: str) -> str:
    return tool.split("_for_")[-1]


def extract_tool_chain(sample):
    conversations = sample.get("conversations", [])
    tools = []

    for turn in conversations:
        if turn.get("from") == "assistant":
            m = re.search(r"Action:\s*([a-zA-Z0-9_]+)", turn.get("value", ""))
            if m:
                t = m.group(1)
                if valid_tool(t):
                    tools.append(t)

    return list(dict.fromkeys(tools))


def load_simple_scenes():
    dataset = load_dataset(
        "Yhyu13/ToolBench_toolllama_G123_dfs",
        split="train"
    )

    scenes = []
    for sample in dataset:
        tools = extract_tool_chain(sample)
        if len(tools) >= MIN_TOOLS_PER_SCENE:
            scenes.append(tools)

    print(f"✓ Parsed {len(scenes)} simple scenes")
    return scenes


# ============================================================
# 3. Service-level 聚合
# ============================================================

def build_service_profiles(scenes):
    """构建每个service的工具和使用统计"""
    service_data = defaultdict(lambda: {
        "tools": set(),
        "scene_count": 0,
        "scenes_using": []
    })

    for idx, scene in enumerate(scenes):
        scene_services = defaultdict(list)
        for tool in scene:
            service = tool_service(tool)
            scene_services[service].append(tool)

        for service, tools in scene_services.items():
            service_data[service]["tools"].update(tools)
            service_data[service]["scene_count"] += 1
            service_data[service]["scenes_using"].append(idx)

    # 过滤小service
    filtered = {}
    for service, data in service_data.items():
        if (len(data["tools"]) >= MIN_TOOLS_PER_SERVICE and
                data["scene_count"] >= MIN_SERVICE_SUPPORT):
            filtered[service] = data

    print(f"✓ Built profiles for {len(filtered)} services")
    print(f"  (filtered out {len(service_data) - len(filtered)} small services)")

    return filtered


# ============================================================
# 4. 主题推断
# ============================================================

def infer_service_theme(service_name, tools):
    """为service推断主题"""
    scores = Counter()
    matched_tools = set()

    # 检查service名称（高权重）
    service_lower = service_name.lower()
    for theme, kws in THEME_KEYWORDS.items():
        for kw in kws:
            if kw in service_lower:
                scores[theme] += 10
                break

    # 检查工具名称
    for tool in tools:
        name = tool.lower()
        for theme, kws in THEME_KEYWORDS.items():
            for kw in kws:
                if kw in name:
                    scores[theme] += 1
                    matched_tools.add(tool)
                    break

    if not scores:
        return None

    # 计算质量指标
    total_tools = len(tools)
    coverage = len(matched_tools) / total_tools if total_tools > 0 else 0

    top_theme, _ = scores.most_common(1)[0]

    # 计算纯度（只基于工具匹配）
    tool_scores = Counter()
    for tool in tools:
        name = tool.lower()
        for theme, kws in THEME_KEYWORDS.items():
            if any(kw in name for kw in kws):
                tool_scores[theme] += 1
                break

    purity = tool_scores[top_theme] / total_tools if tool_scores else 0

    return top_theme, purity, coverage


# ============================================================
# 5. 构建主题Compounds
# ============================================================

def build_theme_compounds(scenes, service_profiles):
    """
    两阶段聚类，但输出格式保持原样
    """
    # 第一步：按主题分组services
    theme_services = defaultdict(list)

    for service, profile in service_profiles.items():
        theme_result = infer_service_theme(service, list(profile["tools"]))

        if theme_result:
            theme, purity, coverage = theme_result
            if purity >= MIN_THEME_PURITY:
                theme_services[theme].append({
                    "service": service,
                    "tools": profile["tools"],
                    "scenes": profile["scenes_using"],
                    "purity": purity
                })

    # 第二步：为每个主题构建compound
    compounds = []
    compound_counter = defaultdict(int)

    for theme, services in sorted(theme_services.items()):
        if not services:
            continue

        # 合并所有service的工具
        all_tools = set()
        all_scene_indices = set()

        for srv in services:
            all_tools.update(srv["tools"])
            all_scene_indices.update(srv["scenes"])

        # 计算support（使用该主题的场景数）
        support = len(all_scene_indices)

        # 计算整体质量
        matched_counts = Counter()
        total_tools = len(all_tools)

        for tool in all_tools:
            name = tool.lower()
            for t, kws in THEME_KEYWORDS.items():
                if any(kw in name for kw in kws):
                    matched_counts[t] += 1
                    break

        if not matched_counts:
            continue

        purity = matched_counts[theme] / total_tools
        coverage = sum(matched_counts.values()) / total_tools

        # 只保留高质量的compound
        if purity < MIN_THEME_PURITY:
            continue

        compound_counter[theme] += 1

        compounds.append({
            "compound_id": f"{theme}_compound_{compound_counter[theme]:04d}",
            "theme": theme,
            "tools": sorted(all_tools),
            "num_tools": len(all_tools),
            "num_simple_scenes": support,
            "total_support": support,
            "quality_metrics": {
                "purity": round(purity, 2),
                "coverage": round(coverage, 2),
                "matched_tools": sum(matched_counts.values()),
                "total_tools": total_tools,
                "theme_distribution": dict(matched_counts.most_common(5))
            }
        })

    return sorted(compounds, key=lambda x: x["num_simple_scenes"], reverse=True)


# ============================================================
# 6. 主流程
# ============================================================

def main():
    print("=" * 60)
    print("两阶段聚类 (Service → Theme) - 保持原输出格式")
    print("=" * 60)

    scenes = load_simple_scenes()

    print("\n[Stage 1] Building service profiles...")
    service_profiles = build_service_profiles(scenes)

    print("\n[Stage 2] Building theme compounds...")
    compounds = build_theme_compounds(scenes, service_profiles)

    print(f"\n✓ Generated {len(compounds)} theme-based compounds")

    # 统计
    theme_stats = Counter()
    for c in compounds:
        theme_stats[c["theme"]] += 1

    print("\n📈 Theme Distribution:")
    for theme, count in theme_stats.most_common():
        total_tools = sum(c["num_tools"] for c in compounds if c["theme"] == theme)
        print(f"  {theme}: {count} compound(s), {total_tools} tools")

    # 保存结果
    output_file = "toolbench_two_stage_compounds.json"
    with open(output_file, "w") as f:
        json.dump(compounds, f, indent=2)

    print(f"\n✓ Saved to {output_file}")

    # 保存样例
    if compounds:
        sample_file = "sample_two_stage.json"
        with open(sample_file, "w") as f:
            json.dump(compounds[:5], f, indent=2)
        print(f"✓ Saved sample to {sample_file}")


if __name__ == "__main__":
    main()
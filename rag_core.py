# -*- coding: utf-8 -*-

import os
import re
import json
import sys
import jieba
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv


# ===================== 加载环境变量 =====================
APP_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(__file__)
INTERNAL_DIR = getattr(sys, "_MEIPASS", APP_DIR)

load_dotenv(os.path.join(APP_DIR, ".env"))
load_dotenv()


# ===================== 基础配置 =====================
# 知识库文件路径
KNOWLEDGE_FILENAME = "structured_control_theory_knowledge_final.txt"
KNOWLEDGE_PATH = os.path.join(APP_DIR, KNOWLEDGE_FILENAME)
if not os.path.exists(KNOWLEDGE_PATH):
    KNOWLEDGE_PATH = os.path.join(INTERNAL_DIR, KNOWLEDGE_FILENAME)

CHROMA_DB_PATH = os.path.join(APP_DIR, "chroma_db")
COLLECTION_NAME = "control_theory"

LOCAL_EMBEDDING_MODEL_PATH = os.path.join(APP_DIR, "models", "text2vec-base-chinese")
EMBEDDING_MODEL_NAME = (
    LOCAL_EMBEDDING_MODEL_PATH
    if os.path.exists(LOCAL_EMBEDDING_MODEL_PATH)
    else "shibing624/text2vec-base-chinese"
)

# 先多召回，再重排序
TOP_K = 20

# 最终送给大模型的知识条数
FINAL_SOURCE_K = 3

# 是否启用大模型相关性重排。
# 作用：在向量检索和关键词重排序之后，再让模型判断候选知识点是否真正能回答问题。
# 这不是针对固定问题写规则，而是通用的“相关性裁判”，未来新增 TXT 知识点仍然适用。
ENABLE_LLM_RERANK = False

# 送入大模型重排的候选知识点数量。数量越多越稳，但会增加一点调用成本。
RERANK_CANDIDATE_K = 8

# 词面补召回阈值：用于把标题/正文强匹配的知识点补进候选集
LEXICAL_SUPPLEMENT_THRESHOLD = 0.28

# 词面可靠阈值：词面匹配达到这个值，可以作为可靠来源
LEXICAL_RELIABLE_THRESHOLD = 0.34

# 向量强相关阈值：相似度达到该值且有一定词面命中，认为可靠
VECTOR_STRONG_SIMILARITY = 0.58

# 向量中等相关阈值：相似度达到该值但需要更强词面命中
VECTOR_MEDIUM_SIMILARITY = 0.50

# 最高向量相似度过低且词面也不强，则拒答
MIN_TOP_SIMILARITY = 0.45

# 向量结果至少需要一点词面支撑，避免“语义看似相关但实际偏题”
MIN_LEXICAL_FOR_VECTOR = 0.08


# ===================== 明显知识库外问题关键词 =====================
# 注意：这里只过滤明显不适合进入课程知识库检索的问题。
# 是否属于知识库覆盖范围，主要交给 retrieve_knowledge() 判断。
OUT_OF_DOMAIN_KEYWORDS = [
    "天气", "气温", "下雨", "新闻", "股票", "基金", "汇率", "电影",
    "旅游", "酒店", "外卖", "菜谱", "美食", "穿搭", "星座", "八卦",
    "游戏", "娱乐", "今天几号", "现在几点", "上海天气", "北京天气",
    "东京天气", "考试安排", "放假", "快递", "医院", "感冒", "减肥",
    "小说", "作文", "翻译", "图片", "画图", "写诗", "写歌"
]


def is_domain_query(query: str) -> bool:
    """
    初步过滤明显知识库外的问题。

    这里不再依赖固定的“自动控制原理关键词表”判断能不能回答。
    未来知识库扩展到根轨迹、频域分析、补偿器设计等新内容时，
    不需要同步修改这里。

    返回 True：允许进入检索流程。
    返回 False：明显知识库外问题，直接拒答。
    """
    q = query.strip()

    for word in OUT_OF_DOMAIN_KEYWORDS:
        if word in q:
            return False

    return True


# ===================== 通用停用词 =====================
STOP_WORDS = {
    "什么", "怎么", "如何", "一下", "一个", "这个", "那个",
    "请问", "告诉", "解释", "说明", "是否", "可以", "一下子",
    "进行", "需要", "属于", "以及", "如果", "那么", "为什么",
    "的", "了", "是", "和", "与", "或", "在", "中", "对",
    "为", "用", "下", "上", "有", "吗", "呢", "吧", "啊",
    "哪些", "多少", "怎样", "之间",

    # 下面这些是“提问意图词”，只用于判断问题类型，
    # 不应该直接参与知识点匹配，否则容易把无关公式条目召回。
    "公式", "计算", "求", "求解", "表达式", "推导",
    "怎么计算", "怎么算", "如何计算", "如何求",
    "方法", "步骤", "怎么做"
}


# ===================== 全局知识记录 =====================
KNOWLEDGE_RECORDS = []

# 动态知识库术语：系统启动时从 TXT 的章节、条目中自动生成
DYNAMIC_KNOWLEDGE_TERMS = set()


# ===================== 初始化模型与向量库 =====================
print("正在初始化系统...")

embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# 每次运行重建集合，保证 TXT 修改后不会继续用旧数据
try:
    client.delete_collection(name=COLLECTION_NAME)
    print("已删除旧集合")
except Exception:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


# ===================== 解析结构化知识条目 =====================
def parse_knowledge_item(text: str):
    """
    从如下格式中解析编号、章节、主题：
    [K001][时间响应][动态性能指标] 动态性能指标：......

    如果某一行没有严格符合格式，也不会报错，会归为“未分类”。
    """
    pattern = r"^\[(K\d+)\]\[([^\]]+)\]\[([^\]]+)\]\s*(.*)$"
    match = re.match(pattern, text)

    if match:
        kid = match.group(1)
        chapter = match.group(2)
        topic = match.group(3)
        content = match.group(4)
    else:
        kid = None
        chapter = "未分类"
        topic = "未标注"
        content = text

    return kid, chapter, topic, content


def add_term(term: str):
    """
    将术语加入 jieba 词典和动态术语集合。
    """
    term = term.strip()
    if not term:
        return
    if len(term) < 2 and term not in {"ζ", "ω", "π"}:
        return

    jieba.add_word(term)
    DYNAMIC_KNOWLEDGE_TERMS.add(term)


def add_dynamic_terms_to_jieba(chapter: str, topic: str, content: str):
    """
    将知识库中的章节名、条目名和部分专业短语动态加入 jieba 词典。

    作用：
    未来新增知识点时，不需要手动维护 Python 里的关键词表。
    只要 TXT 中写了新的章节或条目，系统启动时会自动加入分词词典。
    """
    add_term(chapter)
    add_term(topic)

    # 对条目名进行简单拆分，增强短问句匹配
    for part in re.split(r"[、，,；;：:/\s]+", topic):
        add_term(part)

    # 从正文中提取类似 Kp、Kv、Ka、ess、G(s)、E(s)、ts 这类符号词
    symbol_terms = re.findall(
        r"[A-Za-z]+_?[A-Za-z]*|[A-Za-z]\([^)]+\)|[ζωπσ%]+",
        content
    )

    for term in symbol_terms:
        add_term(term)


# ===================== 加载知识库 =====================
def load_knowledge_to_db(file_path: str) -> bool:
    """
    将结构化知识库加载到 Chroma 向量数据库。
    同时保存到全局 KNOWLEDGE_RECORDS，用于通用关键词重排序和词面补召回。
    """
    global KNOWLEDGE_RECORDS

    if not os.path.exists(file_path):
        print(f"知识库文件不存在: {file_path}")
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        knowledge_list = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            knowledge_list.append(line)

    if not knowledge_list:
        print("知识库为空，请检查 TXT 文件内容")
        return False

    print(f"共加载 {len(knowledge_list)} 条知识")

    embeddings = embed_model.encode(knowledge_list).tolist()

    ids = []
    metadatas = []
    KNOWLEDGE_RECORDS = []

    for i, item in enumerate(knowledge_list):
        kid, chapter, topic, content = parse_knowledge_item(item)

        # 动态加入分词词典，支持未来扩展知识库
        add_dynamic_terms_to_jieba(chapter, topic, content)

        doc_id = kid if kid else f"id_{i}"

        if doc_id in ids:
            doc_id = f"{doc_id}_{i}"

        ids.append(doc_id)

        metadata = {
            "knowledge_id": kid if kid else f"id_{i}",
            "chapter": chapter,
            "topic": topic
        }

        metadatas.append(metadata)

        KNOWLEDGE_RECORDS.append({
            "id": doc_id,
            "document": item,
            "metadata": metadata
        })

    batch_size = 100

    for i in range(0, len(knowledge_list), batch_size):
        collection.add(
            documents=knowledge_list[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            ids=ids[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )

    print(f"知识库导入完成，共 {len(knowledge_list)} 条")
    print(f"动态术语数量：{len(DYNAMIC_KNOWLEDGE_TERMS)}")
    return True


load_knowledge_to_db(KNOWLEDGE_PATH)


# ===================== 初始化通义千问 =====================
llm_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# ===================== 通用核心词抽取 =====================
def extract_query_terms(query: str):
    """
    通用核心词抽取。

    特点：
    1. 不按 K 编号、不按固定题目写死；
    2. 优先使用 TXT 自动生成的章节名、条目名、专业术语；
    3. 再结合 jieba 分词和符号提取；
    4. 未来新增 TXT 条目后，新的章节和条目也会自动参与分词。
    """
    q = query.strip()
    terms = []

    # 1. 优先匹配知识库动态术语
    # 长术语优先，避免“二阶系统调节时间公式”被拆散
    for term in sorted(DYNAMIC_KNOWLEDGE_TERMS, key=len, reverse=True):
        if len(term) >= 2 and term in q:
            terms.append(term)

    # 2. jieba 分词
    words = jieba.lcut(q)

    for w in words:
        w = w.strip()
        if not w:
            continue
        if w in STOP_WORDS:
            continue
        if len(w) < 2 and w not in {"ζ", "ω", "π"}:
            continue
        terms.append(w)

    # 3. 保留常见英文、符号、函数形式，如 Kp、Kv、Ka、ess、ts、G(s)
    symbol_terms = re.findall(
        r"[A-Za-z]+_?[A-Za-z]*|[A-Za-z]\([^)]+\)|[ζωπσ%]+",
        q
    )

    for s in symbol_terms:
        s = s.strip()
        if not s:
            continue
        if s in STOP_WORDS:
            continue
        terms.append(s)

    return list(dict.fromkeys(terms))


# ===================== 公式特征判断 =====================
def has_formula_feature(document: str) -> bool:
    """
    判断知识条目是否包含通用公式特征。

    注意：
    不再把普通括号 () 当作公式特征，否则几乎所有条目都会被误判为公式条目。
    """
    formula_markers = [
        "=", "≈", "lim", "\\frac", "π", "√", "e^", "∞", "→",
        "σ%", "ζ", "ω", "Kp", "Kv", "Ka", "ess", "ts", "tp", "tr"
    ]

    if any(marker in document for marker in formula_markers):
        return True

    # 匹配 G(s)、E(s)、C(s)、R(s) 这类函数形式
    if re.search(r"[A-Za-z]\([^)]+\)", document):
        return True

    return False


def is_formula_query(query: str) -> bool:
    """
    判断用户是否在询问公式、计算或表达式。
    """
    formula_query_words = [
        "公式", "怎么算", "怎么计算", "计算", "求", "表达式",
        "如何求", "如何计算", "推导", "求解"
    ]

    return any(word in query for word in formula_query_words)


def is_condition_query(query: str) -> bool:
    """
    判断用户是否在询问适用条件、前提或范围。
    """
    condition_words = ["条件", "适用", "前提", "什么时候", "场景", "限制", "范围"]
    return any(word in query for word in condition_words)


def is_method_query(query: str) -> bool:
    """
    判断用户是否在询问方法、判断规则或步骤。
    """
    method_words = ["怎么判断", "如何判断", "判据", "规则", "方法", "步骤", "怎么做", "怎么画"]
    return any(word in query for word in method_words)


# ===================== 通用关键词匹配得分 =====================
def lexical_score(query: str, document: str, metadata: dict) -> float:
    """
    通用关键词重排序得分。

    设计原则：
    1. 不按 K 编号加权；
    2. 不针对某个固定示例问题加权；
    3. 重点强化“章节、条目标题、正文”和用户问题的匹配；
    4. 公式类问题优先含公式的条目；
    5. 判断/方法类问题优先含判据、方法、规则、步骤的条目；
    6. 未来扩展 TXT 时，不需要修改这里。
    """
    q = query.strip()
    doc = document
    topic = metadata.get("topic", "")
    chapter = metadata.get("chapter", "")

    score = 0.0
    terms = extract_query_terms(q)

    # 1. 章节和条目标题整体命中，权重较高
    if topic and topic in q:
        score += 0.45

    if chapter and chapter in q:
        score += 0.12

    # 2. 用户问题关键词与章节、条目、正文匹配
    matched_terms = []

    for term in terms:
        if not term:
            continue

        matched = False

        if term in topic:
            score += 0.18
            matched = True

        if term in chapter:
            score += 0.08
            matched = True

        if term in doc:
            score += 0.06
            matched = True

        if matched:
            matched_terms.append(term)

    # 3. 关键词覆盖率：问题里的有效词有多少能在该知识条目中找到
    useful_terms = [t for t in terms if len(t) >= 2 and t not in STOP_WORDS]

    if useful_terms:
        coverage = len(set(matched_terms)) / max(len(set(useful_terms)), 1)
        score += min(coverage * 0.4, 0.4)

    # 4. 公式类问题：优先选择“已经命中专业词”的含公式条目
    # 注意：不能只因为条目里有公式就加分，否则
    # “上升时间怎么计算”可能误召回“稳态误差计算前提”等无关公式条目。
    has_real_term_match = len(set(matched_terms)) > 0

    if is_formula_query(q):
        if has_real_term_match and has_formula_feature(doc):
            score += 0.25
        if has_real_term_match and ("公式" in topic or "计算" in topic or "系数" in topic):
            score += 0.12

    # 5. 条件类问题：优先选择带条件说明或适用前提的条目
    if is_condition_query(q):
        if "条件说明" in doc or "适用" in doc or "前提" in doc:
            score += 0.18

    # 6. 判断/方法类问题：优先选择含判据、判断、方法、规则的条目
    if is_method_query(q):
        if any(word in topic for word in ["判据", "判断", "方法", "规则", "步骤"]):
            score += 0.18
        if any(word in doc for word in ["判据", "判断", "方法", "规则", "步骤"]):
            score += 0.10

    # 7. 知识条目本身带条件说明，略微加分
    if "【条件说明：" in doc:
        score += 0.03

    return round(score, 4)


# ===================== 检索函数 =====================
def retrieve_knowledge(query: str):
    """
    检索知识库，返回相关知识条目。

    流程：
    1. 向量检索召回 TOP_K 条；
    2. 对召回结果计算通用关键词匹配得分；
    3. 对全知识库做轻量词面补召回，避免短问题漏召回关键条目；
    4. 可靠性过滤，避免把语义相近但实际偏题的条目送给模型；
    5. 按综合得分排序，返回前 FINAL_SOURCE_K 条。
    """
    query_embedding = embed_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "distances", "metadatas"]
    )

    docs = results["documents"][0]
    dists = results["distances"][0]
    metas = results["metadatas"][0]

    candidates = {}

    # 1. 向量召回候选
    for doc, dis, meta in zip(docs, dists, metas):
        dis = float(dis)
        similarity = round(1 - dis, 4)
        lex = lexical_score(query, doc, meta)

        # 综合得分：向量相似度 + 通用关键词匹配得分
        score = round(similarity + lex, 4)

        kid = meta.get("knowledge_id", doc[:20])

        candidates[kid] = {
            "document": doc,
            "distance": dis,
            "similarity": similarity,
            "lexical_score": lex,
            "score": score,
            "metadata": meta,
            "source_type": "vector"
        }

    top_vector_similarity = max(
        [item["similarity"] for item in candidates.values()],
        default=0
    )

    # 2. 全知识库词面补召回
    for record in KNOWLEDGE_RECORDS:
        doc = record["document"]
        meta = record["metadata"]
        kid = meta.get("knowledge_id", record["id"])

        lex = lexical_score(query, doc, meta)

        if lex >= LEXICAL_SUPPLEMENT_THRESHOLD:
            item = {
                "document": doc,
                "distance": None,
                "similarity": None,
                "lexical_score": lex,
                # 词面补召回没有真实向量相似度，因此给一个基础分
                "score": round(0.45 + lex, 4),
                "metadata": meta,
                "source_type": "lexical"
            }

            if kid not in candidates or item["score"] > candidates[kid]["score"]:
                candidates[kid] = item

    if not candidates:
        return []

    # 3. 可靠性过滤
    reliable = []

    for item in candidates.values():
        sim = item.get("similarity")
        lex = item.get("lexical_score", 0)

        # 向量强相关：必须至少有一点词面支撑，避免根轨迹误召回劳斯判据这种情况
        vector_strong_ok = (
            sim is not None
            and sim >= VECTOR_STRONG_SIMILARITY
            and lex >= MIN_LEXICAL_FOR_VECTOR
        )

        # 向量中等相关：必须有更强词面匹配
        vector_medium_ok = (
            sim is not None
            and sim >= VECTOR_MEDIUM_SIMILARITY
            and lex >= 0.18
        )

        # 词面强相关：标题/正文匹配足够强
        lexical_ok = lex >= LEXICAL_RELIABLE_THRESHOLD

        if vector_strong_ok or vector_medium_ok or lexical_ok:
            reliable.append(item)

    if not reliable:
        return []

    max_lex = max([item.get("lexical_score", 0) for item in reliable], default=0)

    # 最高向量分很低，且没有强词面匹配，则拒答
    if top_vector_similarity < MIN_TOP_SIMILARITY and max_lex < LEXICAL_RELIABLE_THRESHOLD:
        return []

    # 4. 综合排序
    reliable = sorted(
        reliable,
        key=lambda x: x["score"],
        reverse=True
    )

    # 5. 来源差距过滤
    # 只保留与第一名差距不大的知识点，避免参考知识点里混入弱相关条目。
    # 例如“上升时间怎么计算”不应混入“稳态误差计算前提”“劳斯表首列零元素”。
    if not reliable:
        return []

    top_score = reliable[0]["score"]
    top_lex = reliable[0].get("lexical_score", 0)

    filtered = []
    for item in reliable:
        item_score = item.get("score", 0)
        item_lex = item.get("lexical_score", 0)

        close_to_top = item_score >= top_score - 0.25
        strong_lex = item_lex >= max(top_lex * 0.60, LEXICAL_RELIABLE_THRESHOLD)

        if close_to_top or strong_lex:
            filtered.append(item)

    preliminary_sources = filtered[:RERANK_CANDIDATE_K]

    # 6. 可选 LLM 相关性重排
    # 最终在线版本默认关闭该步骤，避免检索阶段额外调用一次大模型。
    # if ENABLE_LLM_RERANK and preliminary_sources:
    #     reranked_sources = llm_rerank_sources(query, preliminary_sources, FINAL_SOURCE_K)
    #     if reranked_sources is not None:
    #         return reranked_sources

    return preliminary_sources[:FINAL_SOURCE_K]


def llm_rerank_sources(query: str, candidates: list, top_n: int = 3):
    """
    使用大模型对候选知识点进行相关性重排。

    为什么需要这一层：
    1. 向量检索擅长召回语义相近内容，但有时会召回“看起来相关、实际不贴合问题”的知识点；
    2. 关键词重排序能提高精度，但仍然难以完全理解用户问题的核心诉求；
    3. 大模型重排只负责判断“候选知识点是否能支撑回答”，不负责编造新知识。

    这一步是通用机制，不绑定具体 K 编号、不绑定固定题目。
    未来 TXT 新增知识点后，只要被召回为候选，也可以自动参与重排。

    返回：
    - list：筛选后的知识点列表；
    - []：模型认为没有可靠候选；
    - None：重排失败，调用方使用原排序兜底。
    """
    if not candidates:
        return []

    # 如果候选只有 1 条，且已经通过前面可靠性过滤，就不再额外调用模型。
    if len(candidates) == 1:
        return candidates[:top_n]

    candidate_payload = []
    id_to_item = {}

    for idx, item in enumerate(candidates, 1):
        meta = item.get("metadata", {})
        cid = f"C{idx}"
        id_to_item[cid] = item

        doc = item.get("document", "")
        # 控制长度，避免 prompt 过长；重排只需要判断相关性，不需要完整展开。
        if len(doc) > 360:
            doc = doc[:360] + "..."

        candidate_payload.append({
            "id": cid,
            "chapter": meta.get("chapter", ""),
            "topic": meta.get("topic", ""),
            "content": doc
        })

    rerank_prompt = f"""你是知识库检索结果的相关性评估器。
请判断下面候选知识点是否能够直接支撑回答用户问题。

用户问题：{query}

候选知识点：
{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}

要求：
1. 只选择能够直接回答或明显支撑回答用户问题的知识点。
2. 如果知识点只是同一课程领域但不能回答该问题，不要选择。
3. 最多选择 {top_n} 条。
4. 如果没有可靠知识点，selected 返回空数组。
5. 只输出 JSON，不要输出解释性文字。

输出格式：
{{"selected": ["C1", "C2"]}}
"""

    try:
        response = llm_client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {
                    "role": "system",
                    "content": "你是严格的知识库检索相关性评估器，只输出 JSON。"
                },
                {
                    "role": "user",
                    "content": rerank_prompt
                }
            ],
            temperature=0,
            max_tokens=200
        )

        raw = response.choices[0].message.content.strip()

        # 兼容模型可能包裹 ```json 的情况
        raw = re.sub(r"^```json", "", raw).strip()
        raw = re.sub(r"^```", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()

        data = json.loads(raw)
        selected_ids = data.get("selected", [])

        if not isinstance(selected_ids, list):
            return None

        selected = []
        for sid in selected_ids:
            if sid in id_to_item:
                selected.append(id_to_item[sid])

        return selected[:top_n]

    except Exception as e:
        print(f"LLM 重排失败，使用原检索排序：{e}")
        return None


def is_insufficient_answer(answer: str) -> bool:
    """
    判断模型是否认为知识库依据不足。
    若依据不足，则前端不应继续显示那些弱相关来源，避免误导用户。
    """
    markers = [
        "知识库依据不足",
        "未检索到",
        "无法基于当前知识库",
        "无法基于本知识库",
        "没有足够依据",
        "依据不足"
    ]

    return any(marker in answer for marker in markers)


# ===================== RAG 问答 =====================
def rag_qa(query: str, return_sources=True):
    """
    RAG 问答核心函数。
    """
    query = query.strip()

    if not query:
        answer = "请输入有效问题。"
        return (answer, []) if return_sources else answer

    # 这里只过滤明显无关问题，其余交给知识库检索判断是否能回答
    if not is_domain_query(query):
        answer = (
            "当前问题明显超出本知识库的使用范围，"
            "因此无法基于当前知识库给出可靠回答。"
        )
        return (answer, []) if return_sources else answer

    source_infos = retrieve_knowledge(query)

    if not source_infos:
        answer = (
            "当前知识库未检索到与该问题高度相关的可靠知识点，"
            "因此无法基于当前知识库给出可靠回答。"
            "建议补充相关知识点后再查询。"
        )
        return (answer, []) if return_sources else answer

    source_docs = [item["document"] for item in source_infos]
    context = "\n".join(source_docs)

    prompt = f"""你是一位课程知识库问答助手。请严格根据下面的知识回答问题，不要编造知识库中没有的信息。

【知识库检索结果】
{context}

【用户问题】
{query}

【回答原则】
1. 必须优先依据“知识库检索结果”回答。
2. 如果检索结果中已经包含用户问题对应的定义、公式、方法或判据，必须优先使用该条知识，不要说“知识库依据不足”。
3. 如果知识库中确实没有足够依据，请明确说明“知识库依据不足”，不要强行回答。
4. 不要在答案中输出“严格依据知识库”“根据知识库Kxxx”这类开场白，直接回答问题本身。
5. 回答要简洁、准确，适合课程问答系统展示。
6. 不要简单拼接知识库原文，要围绕用户问题进行归纳整合。
7.如果用户询问定义类问题，且检索知识点中包含对应公式，应同时给出定义和公式，不要只给文字解释。

【排版要求】
请根据用户问题类型选择最合适的回答结构，不要所有问题都机械套用同一种格式。

1. 如果用户问“是什么、定义、含义”：
   可使用结构：
   简要答案：
   适用条件：
   补充说明：

2. 如果用户问“公式、怎么算、计算、求、表达式”：
   可使用结构：
   简要答案：
   常用公式：
   适用条件：
   补充说明：

3. 如果用户问“有什么影响、有什么作用、区别、关系”：
   可使用结构：
   简要答案：
   分类说明：
   补充说明：

4. 如果用户问“怎么判断、判据、规则、方法”：
   可使用结构：
   简要答案：
   判断方法：
   适用条件：
   补充说明：

5. 如果某一栏没有必要，请直接省略，不要为了凑格式强行生成。
6. 回答应让用户先看到核心结论，再看条件和说明。

【通用专业约束】
1. 数学公式必须使用标准 LaTeX 格式。
2. 回答正文不要使用竖线作为分隔符。
3. 不要使用方括号代替 LaTeX 花括号。
4. 禁止使用任何星号，不要用星号做加粗或列表。
5. 涉及公式、定理、判据或方法时，必须说明适用前提与约束条件。
6. 公式必须来自“知识库检索结果”，不得自行补充知识库中没有的公式。
7. 若检索结果中给出了多个近似公式或不同条件下的公式，必须说明它们分别对应的条件，不能混用。
8. 不要把某一条知识点的特殊适用条件扩大成全部情况，也不要把通用公式缩窄成只适用于某一种系统型别。
9. 如果问题是综合型问题，例如“是什么并如何减小”“区别是什么”“有什么影响”，应整合多个相关知识点进行回答，而不是只复述其中一条。
10. 若检索结果不足以支持回答，应明确说明“知识库依据不足”。

请直接给出答案："""

    try:
        response = llm_client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {
                    "role": "system",
                    "content": "你是严谨的课程知识库问答助手，回答必须依据提供的知识库内容。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.01,
            max_tokens=800
        )

        answer = response.choices[0].message.content.strip()

        # 如果模型最终判断依据不足，就不要把弱相关来源继续展示给用户
        if is_insufficient_answer(answer):
            return (answer, []) if return_sources else answer

        if return_sources:
            return answer, source_infos

        return answer

    except Exception as e:
        error_msg = f"调用失败：{str(e)}"
        return (error_msg, source_infos) if return_sources else error_msg


# ===================== 纯 LLM 对比 =====================
def direct_llm_qa(query: str):
    """
    纯大模型问答，不使用知识库。
    用于后续 RAG 与直接问 LLM 的对比实验。
    """
    prompt = f"""请直接回答下面的问题，要求准确、简洁：

问题：{query}
"""

    try:
        response = llm_client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"调用失败：{str(e)}"


# ===================== 命令行测试 =====================
if __name__ == "__main__":
    test_queries = [
        "二阶系统调节时间公式是什么？",
        "什么是超调量？",
        "劳斯判据怎么判断不稳定根？",
        "一阶系统和二阶系统的响应区别？",
        "阻尼比ζ对系统有什么影响？",
        "稳态误差怎么算？",
        "什么是稳态误差？如何减小？",
        "根轨迹怎么画？",
        "今天上海天气怎么样？"
    ]

    for q in test_queries:
        print("\n" + "=" * 70)
        print(f"问题: {q}")

        answer, sources = rag_qa(q, return_sources=True)

        print(f"\n回答:\n{answer}")

        print("\n来源:")
        if not sources:
            print("无可靠来源")
        else:
            for i, item in enumerate(sources, 1):
                meta = item.get("metadata", {})
                print(
                    f"{i}. {meta.get('knowledge_id', '')} "
                    f"{meta.get('chapter', '')} "
                    f"{meta.get('topic', '')} "
                    f"相似度: {item.get('similarity')} "
                    f"关键词得分: {item.get('lexical_score')} "
                    f"综合得分: {item.get('score')}"
                )
                print(f"   {item['document'][:220]}...")

# -*- coding: utf-8 -*-

"""
知识库检索质量评估脚本

作用：
1. 只评估 retrieve_knowledge() 的检索效果，不调用大模型生成回答。
2. 统计 Top1 Accuracy、Recall@3、Recall@5 和平均检索耗时。
3. 输出失败样例，便于分析知识库或检索规则的问题。
4. 生成 retrieval_eval_report.txt，作为检索实验记录。
"""

import time
from collections import defaultdict

import rag_core


# 为了评估 Recall@5，临时让 retrieve_knowledge() 最多返回 5 条。
# 这不会修改 rag_core.py 文件本身，只影响当前脚本运行过程。
rag_core.FINAL_SOURCE_K = 5
rag_core.RERANK_CANDIDATE_K = 8
rag_core.ENABLE_LLM_RERANK = False


TEST_CASES = [
    # 时间响应：定义类
    {
        "query": "什么是时域分析法？",
        "target_id": "K001",
        "category": "定义类"
    },
    {
        "query": "自动控制原理里常见的典型输入信号有哪些？",
        "target_id": "K002",
        "category": "定义类"
    },
    {
        "query": "动态性能指标包括哪些内容？",
        "target_id": "K003",
        "category": "定义类"
    },
    {
        "query": "什么是上升时间？",
        "target_id": "K004",
        "category": "定义类"
    },
    {
        "query": "什么是峰值时间？",
        "target_id": "K005",
        "category": "定义类"
    },
    {
        "query": "超调量的定义是什么？",
        "target_id": "K006",
        "category": "定义类"
    },
    {
        "query": "调节时间是什么意思？",
        "target_id": "K007",
        "category": "定义类"
    },

    # 时间响应：公式类
    {
        "query": "一阶系统的传递函数怎么写？",
        "target_id": "K009",
        "category": "公式类"
    },
    {
        "query": "一阶系统单位阶跃响应公式是什么？",
        "target_id": "K011",
        "category": "公式类"
    },
    {
        "query": "一阶系统调节时间怎么算？",
        "target_id": "K012",
        "category": "公式类"
    },
    {
        "query": "一阶系统单位斜坡输入下的稳态误差是多少？",
        "target_id": "K014",
        "category": "公式类"
    },
    {
        "query": "标准二阶系统闭环传递函数是什么？",
        "target_id": "K016",
        "category": "公式类"
    },
    {
        "query": "二阶系统阻尼振荡频率公式是什么？",
        "target_id": "K017",
        "category": "公式类"
    },
    {
        "query": "欠阻尼二阶系统的峰值时间怎么计算？",
        "target_id": "K020",
        "category": "公式类"
    },
    {
        "query": "二阶系统超调量公式是什么？",
        "target_id": "K021",
        "category": "公式类"
    },
    {
        "query": "二阶系统调节时间公式是什么？",
        "target_id": "K023",
        "category": "公式类"
    },

    # 时间响应：综合关系类
    {
        "query": "阻尼比ζ对二阶系统响应有什么影响？",
        "target_id": "K019",
        "category": "综合类"
    },
    {
        "query": "二阶系统中ζ越大超调量会怎么变化？",
        "target_id": "K022",
        "category": "综合类"
    },
    {
        "query": "怎样改善二阶系统的动态性能？",
        "target_id": "K027",
        "category": "方法类"
    },
    {
        "query": "比例微分控制有什么作用？",
        "target_id": "K028",
        "category": "方法类"
    },
    {
        "query": "测速反馈控制可以改善什么性能？",
        "target_id": "K029",
        "category": "方法类"
    },
    {
        "query": "一阶系统和二阶系统响应有什么区别？",
        "target_id": "K062",
        "category": "区别类"
    },

    # 稳定性
    {
        "query": "控制系统稳定性的定义是什么？",
        "target_id": "K034",
        "category": "定义类"
    },
    {
        "query": "线性定常系统稳定的充要条件是什么？",
        "target_id": "K035",
        "category": "判据类"
    },
    {
        "query": "特征方程系数全为正能不能保证系统稳定？",
        "target_id": "K036",
        "category": "判据类"
    },
    {
        "query": "三阶系统稳定条件是什么？",
        "target_id": "K037",
        "category": "判据类"
    },
    {
        "query": "劳斯稳定判据怎么判断系统稳定？",
        "target_id": "K038",
        "category": "判据类"
    },
    {
        "query": "劳斯表首列符号变化说明什么？",
        "target_id": "K039",
        "category": "判据类"
    },
    {
        "query": "劳斯表首列出现零元素怎么办？",
        "target_id": "K040",
        "category": "判据类"
    },
    {
        "query": "劳斯表出现全零行代表什么情况？",
        "target_id": "K041",
        "category": "判据类"
    },
    {
        "query": "什么是临界稳定？",
        "target_id": "K042",
        "category": "定义类"
    },

    # 稳态误差
    {
        "query": "什么是稳态误差？",
        "target_id": "K008",
        "category": "定义类"
    },
    {
        "query": "计算稳态误差需要满足什么前提？",
        "target_id": "K044",
        "category": "条件类"
    },
    {
        "query": "什么是系统型别？",
        "target_id": "K045",
        "category": "定义类"
    },
    {
        "query": "静态位置误差系数Kp是什么？",
        "target_id": "K046",
        "category": "公式类"
    },
    {
        "query": "静态速度误差系数Kv是什么？",
        "target_id": "K047",
        "category": "公式类"
    },
    {
        "query": "静态加速度误差系数Ka是什么？",
        "target_id": "K048",
        "category": "公式类"
    },
    {
        "query": "0型系统对阶跃输入的稳态误差怎么算？",
        "target_id": "K049",
        "category": "公式类"
    },
    {
        "query": "I型系统对斜坡输入的稳态误差是多少？",
        "target_id": "K050",
        "category": "公式类"
    },
    {
        "query": "II型系统对加速度输入的稳态误差是多少？",
        "target_id": "K051",
        "category": "公式类"
    },
    {
        "query": "增大开环增益能减小稳态误差吗？",
        "target_id": "K055",
        "category": "方法类"
    },
    {
        "query": "积分环节为什么能改善稳态误差？",
        "target_id": "K056",
        "category": "方法类"
    },
    {
        "query": "PI控制对稳态误差有什么作用？",
        "target_id": "K057",
        "category": "方法类"
    },
    {
        "query": "前馈补偿如何减小稳态误差？",
        "target_id": "K058",
        "category": "方法类"
    },
    {
        "query": "减小稳态误差有哪些方法？",
        "target_id": "K063",
        "category": "方法类"
    },
]


def get_retrieved_ids(results):
    ids = []
    for item in results:
        metadata = item.get("metadata", {})
        kid = metadata.get("knowledge_id", "")
        ids.append(kid)
    return ids


def format_sources(results):
    lines = []
    for rank, item in enumerate(results, 1):
        metadata = item.get("metadata", {})
        lines.append(
            f"{rank}. {metadata.get('knowledge_id', '')} "
            f"{metadata.get('chapter', '')} "
            f"{metadata.get('topic', '')} "
            f"score={item.get('score')} "
            f"sim={item.get('similarity')} "
            f"lex={item.get('lexical_score')}"
        )
    return "\n".join(lines)


def evaluate():
    total = len(TEST_CASES)
    top1_hit = 0
    recall3_hit = 0
    recall5_hit = 0
    total_time = 0.0

    category_stats = defaultdict(lambda: {
        "total": 0,
        "top1": 0,
        "recall3": 0,
        "recall5": 0,
        "time": 0.0
    })

    failed_cases = []
    detail_lines = []

    for index, case in enumerate(TEST_CASES, 1):
        query = case["query"]
        target_id = case["target_id"]
        category = case["category"]

        start_time = time.perf_counter()
        results = rag_core.retrieve_knowledge(query)
        elapsed = time.perf_counter() - start_time

        retrieved_ids = get_retrieved_ids(results)

        hit_top1 = len(retrieved_ids) >= 1 and retrieved_ids[0] == target_id
        hit_recall3 = target_id in retrieved_ids[:3]
        hit_recall5 = target_id in retrieved_ids[:5]

        top1_hit += int(hit_top1)
        recall3_hit += int(hit_recall3)
        recall5_hit += int(hit_recall5)
        total_time += elapsed

        category_stats[category]["total"] += 1
        category_stats[category]["top1"] += int(hit_top1)
        category_stats[category]["recall3"] += int(hit_recall3)
        category_stats[category]["recall5"] += int(hit_recall5)
        category_stats[category]["time"] += elapsed

        status = "PASS" if hit_recall5 else "FAIL"

        detail_lines.append(
            f"[{index:02d}] {status} | {category} | target={target_id} | "
            f"top_ids={retrieved_ids} | time={elapsed:.4f}s\n"
            f"query: {query}\n"
            f"{format_sources(results)}\n"
        )

        if not hit_recall5:
            failed_cases.append({
                "index": index,
                "query": query,
                "target_id": target_id,
                "category": category,
                "retrieved_ids": retrieved_ids,
                "sources": format_sources(results)
            })

    report_lines = []

    report_lines.append("知识库检索质量评估报告")
    report_lines.append("=" * 60)
    report_lines.append(f"测试样例数: {total}")
    report_lines.append(f"Top1 Accuracy: {top1_hit / total:.2%} ({top1_hit}/{total})")
    report_lines.append(f"Recall@3: {recall3_hit / total:.2%} ({recall3_hit}/{total})")
    report_lines.append(f"Recall@5: {recall5_hit / total:.2%} ({recall5_hit}/{total})")
    report_lines.append(f"平均检索耗时: {total_time / total:.4f} 秒")
    report_lines.append("")

    report_lines.append("按问题类型统计")
    report_lines.append("-" * 60)
    for category, stats in category_stats.items():
        n = stats["total"]
        report_lines.append(
            f"{category}: "
            f"Top1={stats['top1'] / n:.2%}, "
            f"Recall@3={stats['recall3'] / n:.2%}, "
            f"Recall@5={stats['recall5'] / n:.2%}, "
            f"AvgTime={stats['time'] / n:.4f}s, "
            f"N={n}"
        )
    report_lines.append("")

    report_lines.append("失败样例")
    report_lines.append("-" * 60)
    if not failed_cases:
        report_lines.append("无 Recall@5 失败样例。")
    else:
        for item in failed_cases:
            report_lines.append(
                f"[{item['index']:02d}] {item['category']} | target={item['target_id']}\n"
                f"query: {item['query']}\n"
                f"retrieved_ids: {item['retrieved_ids']}\n"
                f"{item['sources']}\n"
            )

    report_lines.append("")
    report_lines.append("逐题明细")
    report_lines.append("-" * 60)
    report_lines.extend(detail_lines)

    report = "\n".join(report_lines)

    print(report)

    with open("retrieval_eval_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n评估报告已保存到 retrieval_eval_report.txt")


if __name__ == "__main__":
    evaluate()

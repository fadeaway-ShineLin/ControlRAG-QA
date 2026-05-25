# -*- coding: utf-8 -*-
"""
对比实验：RAG系统 vs 纯大模型（无知识库）
按问题类别记录回答内容、响应时间和引用数量。
"""

import time
import csv
from rag_core import rag_qa, direct_llm_qa

# ===================== 测试问题集 =====================
TEST_QUESTIONS = [
    ("定义", "什么是超调量？"),
	("定义", "什么是临界阻尼？"),
	("定义", "什么是临界稳定？"),	
	("定义", "什么是上升时间？"),
	
    ("公式", "二阶系统调节时间公式是什么？"),
	("公式", "二阶系统峰值时间公式是什么？"),
	("公式", "静态速度误差系数Kv的公式是什么？"),  
	("公式", "静态加速度误差系数Ka的公式是什么？"),	
	
    ("判据", "劳斯判据怎么判断不稳定根？"),
	("判据", "劳斯表出现全零行时怎么办？"),
	("判据", "三阶系统稳定的条件是什么？"),
	("判据", "特征方程系数全正能否判断系统稳定？"),
	
    ("区别", "一阶系统和二阶系统的响应区别？"),
	("比较", "比例微分控制和测速反馈控制的区别？"),
    ("参数影响", "阻尼比ζ对系统有什么影响？"),
	
    ("方法", "如何改善二阶系统的动态性能？"),
	("方法", "积分环节在控制系统中有什么作用？"),
    ("综合", "什么是稳态误差？如何减小？"),
	("综合", "增大开环增益对系统有什么影响？"),
	("型别", "0型系统的稳态误差特性是什么？"),  
]

def run_comparison():
    print("=" * 80)
    print("对比实验：RAG增强系统 vs 纯大模型（无知识库）")
    print("=" * 80)

    results = []

    for idx, (category, q) in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{idx}/{len(TEST_QUESTIONS)}] [{category}] {q}")
        print("-" * 50)

        # RAG
        start = time.time()
        rag_answer, sources = rag_qa(q, return_sources=True)
        rag_time = time.time() - start

        # 纯LLM
        start = time.time()
        llm_answer = direct_llm_qa(q)
        llm_time = time.time() - start

        source_count = len(sources) if sources else 0

        results.append({
            "类别": category,
            "问题": q,
            "RAG回答": rag_answer,
            "RAG引用知识点数": source_count,
            "RAG耗时(秒)": round(rag_time, 2),
            "纯LLM回答": llm_answer,
            "纯LLM耗时(秒)": round(llm_time, 2),
        })

        print(f"RAG回答: {rag_answer[:100]}...")
        print(f"纯LLM回答: {llm_answer[:100]}...")
        print(f"RAG耗时: {rag_time:.2f}s | 纯LLM耗时: {llm_time:.2f}s")

    # 控制台输出汇总表
    print("\n\n" + "=" * 80)
    print("汇总结果")
    print("=" * 80)

    print("\n| 序号 | 类别 | 问题 | RAG回答（前60字） | 纯LLM回答（前60字） | RAG引用数 | RAG耗时(s) | 纯LLM耗时(s) |")
    print("|------|------|------|-------------------|---------------------|-----------|------------|--------------|")
    for i, r in enumerate(results, 1):
        rag_preview = r["RAG回答"].replace("\n", " ")[:60]
        llm_preview = r["纯LLM回答"].replace("\n", " ")[:60]
        print(f"| {i} | {r['类别']} | {r['问题']} | {rag_preview} | {llm_preview} | {r['RAG引用知识点数']} | {r['RAG耗时(秒)']} | {r['纯LLM耗时(秒)']} |")

    # 保存 CSV
    with open("comparison_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n\n详细结果已保存到 comparison_results.csv")

if __name__ == "__main__":
    run_comparison()
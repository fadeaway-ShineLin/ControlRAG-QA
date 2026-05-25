# 自动控制原理 RAG 问答系统

本项目是一个面向《自动控制原理》课程的检索增强生成（Retrieval-Augmented Generation, RAG）问答系统。系统先从结构化课程知识库中检索相关知识条目，再将检索结果、用户问题和回答规则共同构造成 Prompt，调用 `qwen-plus` 生成答案，并在前端展示回答内容和依据来源。

![系统界面示例](docs/system_demo.png)

## 快速开始

```bash
git clone https://github.com/fadeaway-Shinelinn/ControlRAG-QA.git
cd ControlRAG-QA
pip install -r requirements.txt
copy .env.example .env
```

然后打开 `.env` 文件，填写自己的 DashScope API Key：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

配置完成后，再启动 RAG 问答系统：

```bash
python rag_demo.py
```

启动后在浏览器中打开终端显示的 Gradio 地址，即可使用系统。

## 项目功能

- 构建包含 71 条知识点的自动控制原理结构化知识库
- 支持时间响应、稳定性判据、稳态误差三个核心模块的课程问答
- 使用 `text2vec-base-chinese` 进行中文文本向量化
- 使用 ChromaDB 存储知识向量并进行语义召回
- 结合 `lexical_score` 关键词重排序、词面补召回和可靠性过滤
- 使用 `qwen-plus` 生成结构化回答
- 基于 Gradio 提供 Web 问答界面
- 支持公式显示、示例问题、复制回答和依据来源展示
- 提供知识库检索质量评估和 RAG/纯 LLM 对比实验脚本

## 目录结构

```text
ControlRAG-QA/
├─ rag_core.py                                   # RAG 后端核心逻辑
├─ rag_demo.py                                   # Gradio 问答系统界面
├─ pure_llm_demo.py                              # 纯 LLM 基线问答脚本
├─ eval_knowledge_retrieval.py                   # 知识库检索质量评估脚本
├─ compare_rag_vs_llm.py                         # RAG 与纯 LLM 对比实验脚本
├─ structured_control_theory_knowledge_final.txt # 结构化自动控制原理知识库
├─ requirements.txt                              # Python 依赖
├─ .env.example                                  # 环境变量示例
├─ .gitignore                                    # Git 忽略规则
├─ results/                                      # 实验结果
└─ docs/                                         # 项目说明图片
```

## 环境准备

建议使用 Python 3.10。

```bash
pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，并填写自己的 DashScope API Key：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```


## 运行系统

启动 RAG 问答系统：

```bash
python rag_demo.py
```

启动后在浏览器中打开 Gradio 页面，即可输入自动控制原理相关问题进行测试。

## 运行实验

知识库检索质量评估：

```bash
python eval_knowledge_retrieval.py
```

RAG 系统与纯 LLM 对比实验：

```bash
python compare_rag_vs_llm.py
```

## 实验结果概述

知识库检索质量评估使用 45 个典型课程检索样例：

- Top1 Accuracy：86.67%
- Recall@3：100.00%
- Recall@5：100.00%
- 平均检索耗时：0.0475 秒

RAG 与纯 LLM 对比实验使用 20 个自动控制原理典型问题：

- RAG 平均完整性分：2.95
- 纯 LLM 平均完整性分：2.10
- RAG 满分回答数量：19/20
- 纯 LLM 满分回答数量：4/20
- RAG 平均耗时：4.435 秒
- 纯 LLM 平均耗时：1.973 秒

实验结果表明，RAG 系统相比纯 LLM 响应时间更长，但回答完整性更高，并且能够展示知识来源。

## 当前局限

- 知识库目前主要覆盖时间响应、稳定性判据和稳态误差三个模块
- 尚未覆盖根轨迹、频域分析、系统校正等完整课程内容
- 对比实验样例数量有限，完整性评分存在一定人工主观性
- 暂未实现知识库可视化管理后台

## 后续改进方向

- 扩展知识库覆盖范围
- 引入真实学生提问数据
- 开展多评审盲评和更细致的消融实验
- 开发知识库管理界面，支持教师维护知识条目


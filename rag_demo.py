# -*- coding: utf-8 -*-
import gradio as gr
import re
import time
import traceback
from datetime import datetime
from rag_core import rag_qa


APP_CSS = """
:root {
    --primary: #1f4e79;
    --primary-dark: #173b5c;
    --accent: #2f7d62;
    --bg: #f5f7fb;
    --panel: #ffffff;
    --line: #d9e1ec;
    --text: #1f2937;
    --muted: #64748b;
}

body {
    background: var(--bg);
}

.gradio-container {
    max-width: 1500px !important;
    margin: 0 auto !important;
    color: var(--text);
}

#app-header {
    padding: 12px 18px;
    margin: 6px 0 10px 0;
    background: linear-gradient(135deg, #f8fbff 0%, #eef5f3 100%);
    border: 1px solid var(--line);
    border-radius: 8px;
}

#app-header h1 {
    margin: 0 0 4px 0;
    font-size: 24px;
    line-height: 1.25;
    color: var(--primary-dark);
    letter-spacing: 0;
}

#app-header p {
    margin: 0;
    color: var(--muted);
    font-size: 14px;
}

#chatbot {
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--panel);
}

#side-panel {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 14px;
    background: var(--panel);
}

#side-panel h3 {
    margin-top: 0;
    color: var(--primary-dark);
}

#source-note {
    border-left: 4px solid var(--accent);
    background: #f1f8f5;
    padding: 10px 12px;
    border-radius: 6px;
    color: #24483c;
    font-size: 14px;
}

button.primary {
    background: var(--primary) !important;
    border-color: var(--primary) !important;
}

textarea, input {
    border-radius: 6px !important;
}

#bottom-controls {
    align-items: center;
}

#bottom-controls button {
    min-height: 42px !important;
    padding-top: 8px !important;
    padding-bottom: 8px !important;
}

#copy-status textarea {
    min-height: 42px !important;
    height: 42px !important;
}

#copy-status {
    min-height: 18px !important;
    color: var(--muted);
    font-size: 13px;
    margin-top: 4px !important;
}

#copy-status p {
    margin: 0 !important;
}

#side-panel textarea {
    min-height: 72px !important;
}
"""


# ===================== 修复 LaTeX 公式 =====================
def fix_latex_formulas(text):
    """
    修复模型偶尔生成的 LaTeX 包裹符号。
    """
    text = re.sub(r"\\\[", "$$", text)
    text = re.sub(r"\\\]", "$$", text)

    if "\\frac" in text and "$$" not in text and "$" not in text:
        text = re.sub(r"(\\frac\{[^}]+\}\{[^}]+\})", r"$$\1$$", text)

    return text


# ===================== 格式化答案 =====================
def format_answer(answer_text):
    """
    对模型答案做前端展示优化。
    保持通用性，不再使用固定课程术语表进行高亮。
    """
    if not answer_text:
        return ""

    # 清除星号，避免 Markdown 加粗干扰 LaTeX 和界面
    answer_text = answer_text.replace("*", "")

    # 清除无效 HTML 标签
    answer_text = re.sub(r"<\w+.*?>", "", answer_text)
    answer_text = re.sub(r"</\w+>", "", answer_text)

    # 只删除模型可能生成的多余格式标签，不删除【条件说明】
    extra_labels = [
        "【回答】", "【答案】", "【直接回答】",
        "【参考答案】", "【最终答案】"
    ]

    for label in extra_labels:
        answer_text = answer_text.replace(label, "")

    answer_text = fix_latex_formulas(answer_text)

    answer_text = re.sub(r"\n{3,}", "\n\n", answer_text)

    return answer_text.strip()


# ===================== 格式化参考知识点 =====================
def format_sources(sources):
    """
    格式化参考知识点。
    前端展示知识点编号、章节和条目，便于说明答案依据。
    """
    if not sources:
        return ""

    formatted = []
    seen = set()

    for src in sources:
        if isinstance(src, dict):
            metadata = src.get("metadata", {})

            knowledge_id = metadata.get("knowledge_id", "")
            chapter = metadata.get("chapter", "未分类")
            topic = metadata.get("topic", "未标注")

            # 去重，避免同一个知识点重复显示
            source_key = (chapter, topic)
            if source_key in seen:
                continue
            seen.add(source_key)

            formatted.append(
                f"{len(formatted) + 1}. `{knowledge_id}` {chapter}：{topic}"
            )

        else:
            doc_show = str(src)
            if len(doc_show) > 80:
                doc_show = doc_show[:80] + "..."
            formatted.append(f"{len(formatted) + 1}. {doc_show}")

    if not formatted:
        return ""

    return "\n".join(formatted)


# ===================== 聊天函数 =====================
def chat_function(message, history):
    """
    Gradio 聊天主函数。
    """
    if not message or not message.strip():
        return history

    with open("启动问答系统.log", "a", encoding="utf-8") as log_file:
        log_file.write(f"[{datetime.now()}] receive question: {message}\n")

    try:
        answer, sources = rag_qa(message, return_sources=True)

        formatted_answer = format_answer(answer)
        formatted_sources = format_sources(sources)

        if formatted_sources:
            full_response = (
                f"{formatted_answer}\n\n"
                f"---\n"
                f"### 依据来源\n\n"
                f"{formatted_sources}"
            )
        else:
            full_response = formatted_answer

        history.append((message, full_response))

        with open("启动问答系统.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"[{datetime.now()}] answer finished\n")
    except Exception as e:
        error_text = (
            "系统调用失败，错误信息如下：\n\n"
            f"{type(e).__name__}: {e}\n\n"
            "请查看项目文件夹中的 启动问答系统.log 获取完整错误记录。"
        )
        history.append((message, error_text))
        with open("启动问答系统.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"[{datetime.now()}] answer failed\n")
            log_file.write(traceback.format_exc())
            log_file.write("\n")

    return history


# ===================== 复制全部对话 =====================
def copy_all_conversation(history):
    """
    复制全部对话内容。
    """
    if not history:
        return "暂无对话内容"

    all_text = []

    for q, a in history:
        clean_q = re.sub(r"<.*?>", "", q)
        clean_a = re.sub(r"<.*?>", "", a)

        all_text.append(
            f"【问】{clean_q}\n\n【答】{clean_a}\n\n{'=' * 40}\n"
        )

    full_text = "\n".join(all_text)

    try:
        import pyperclip
        pyperclip.copy(full_text.strip())
        return "已复制全部对话"
    except ImportError:
        return "请安装 pyperclip: pip install pyperclip"
    except Exception as e:
        return f"复制失败：{str(e)}"


def clear_copy_status():
    """
    短暂显示复制结果后清空提示。
    """
    time.sleep(2.5)
    return ""


# ===================== 构建 Gradio 界面 =====================
with gr.Blocks(
    title="自动控制原理知识库增强问答系统",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate"
    ),
    css=APP_CSS
) as demo:

    gr.Markdown(
        """
        # 自动控制原理知识库增强问答系统
        面向公式、判据和条件说明的课程问答原型，回答由结构化知识库检索结果支撑。
        """,
        elem_id="app-header"
    )

    with gr.Row():

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="问答结果",
                height=590,
                show_copy_button=True,
                bubble_full_width=False,
                elem_id="chatbot",
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False}
                ]
            )

            with gr.Row(elem_id="bottom-controls"):
                clear_btn = gr.Button("清空对话", variant="secondary", size="sm")
                copy_all_btn = gr.Button("复制全部对话", variant="secondary", size="sm")
            copy_status = gr.Markdown("", elem_id="copy-status")

        with gr.Column(scale=1, elem_id="side-panel"):
            gr.Markdown("### 系统说明")
            gr.Markdown("""
            - 面向自动控制原理课程问答
            - 基于结构化知识库检索生成
            - 支持公式、条件和来源展示
            - 知识不足时返回拒答提示
            """)

            gr.Markdown(
                "回答下方的“依据来源”展示知识点编号、章节和条目名称，可对应追溯到结构化知识库原始条目。",
                elem_id="source-note"
            )

            gr.Markdown("### 示例问题")

            user_input = gr.Textbox(
                label="请输入你的问题",
                placeholder="例如：二阶系统调节时间公式是什么？",
                lines=2,
                max_lines=2
            )

            submit_btn = gr.Button("提交", variant="primary", size="md")

            example_questions = [
                "二阶系统调节时间公式是什么？",
                "什么是超调量？",
                "劳斯判据怎么判断不稳定根？",
                "一阶系统和二阶系统的响应区别？",
                "阻尼比ζ对系统有什么影响？",            
                "什么是稳态误差？"                          
            ]

            gr.Examples(
                examples=example_questions,
                inputs=user_input,
                label=""
            )

    # ===================== 绑定事件 =====================
    submit_btn.click(
        chat_function,
        inputs=[user_input, chatbot],
        outputs=[chatbot],
        queue=False
    ).then(
        lambda: "",
        outputs=[user_input],
        queue=False
    )

    clear_btn.click(
        lambda: [],
        outputs=[chatbot],
        queue=False
    ).then(
        lambda: "",
        outputs=[user_input],
        queue=False
    )

    copy_all_btn.click(
        copy_all_conversation,
        inputs=[chatbot],
        outputs=[copy_status],
        queue=False
    ).then(
        clear_copy_status,
        outputs=[copy_status],
        queue=False
    )

    user_input.submit(
        chat_function,
        inputs=[user_input, chatbot],
        outputs=[chatbot],
        queue=False
    ).then(
        lambda: "",
        outputs=[user_input],
        queue=False
    )


# ===================== 启动 Web 服务 =====================
if __name__ == "__main__":
    # PyInstaller 打包后，Gradio 的 localhost 自检在部分 Windows 环境下可能误判。
    # 这里跳过该自检，实际服务仍绑定到本机地址并由浏览器访问。
    try:
        import gradio.networking as gradio_networking
        gradio_networking.url_ok = lambda url: True
    except Exception:
        pass

    print("启动 Web 界面...")
    print("   访问地址: http://127.0.0.1:7860")

    demo.launch(
        inbrowser=True,
        server_name="127.0.0.1",
		show_api=False
    )

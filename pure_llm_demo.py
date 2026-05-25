import gradio as gr
import re
from rag_core import direct_llm_qa

# 公式修复函数（和 RAG 系统中类似）
def fix_latex(text):
    # 把 \text{frac}[3] 转换为 \frac{3}{...}
    text = re.sub(r'\\text\{frac\}\[(\d+)\]\(([^)]+)\)', r'\\frac{\1}{\2}', text)
    # 把 s_t 改为 t_s（常见笔误）
    text = re.sub(r's_t', 't_s', text)
    # 把 \text{approx} 替换为 \approx
    text = re.sub(r'\\text\{approx\}', r'\\approx', text)
    # 如果整个字符串没有 $$ 包裹，且包含 \frac，则用 $$ 包裹
    if '\\frac' in text and '$$' not in text:
        text = '$$' + text + '$$'
    return text

def answer_with_fix(question):
    raw = direct_llm_qa(question)
    fixed = fix_latex(raw)
    return fixed

# 创建界面，并启用 LaTeX 渲染（rag_demo一致）
with gr.Blocks(theme=gr.themes.Glass()) as demo:
    gr.Markdown("## 纯大模型回答演示（公式已渲染）")
    with gr.Row():
        question = gr.Textbox(label="请输入问题", lines=2)
        answer = gr.Markdown(label="回答", latex_delimiters=[
            {"left": "$$", "right": "$$", "display": True},
            {"left": "$", "right": "$", "display": False}
        ])
    submit = gr.Button("提问")
    submit.click(answer_with_fix, inputs=question, outputs=answer)

if __name__ == "__main__":
    demo.launch(share=True, server_port=7860)
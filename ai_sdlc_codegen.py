
# ===============================
# 0. Install dependencies
# ===============================
!pip install -q transformers accelerate torch gradio PyPDF2 huggingface_hub

import gradio as gr
import torch
import PyPDF2
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===============================
# 1. Set Hugging Face Token
# ===============================
# Replace with your OWN token from https://huggingface.co/settings/tokens
HF_TOKEN = "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"

# ===============================
# 2. Load Granite Model (PRIVATE)
# ===============================
model_name = "ibm-granite/granite-3.2-2b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=HF_TOKEN,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===============================
# 3. Helper Functions
# ===============================
def generate_response(prompt, max_length=400):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def extract_text_from_pdf(pdf_file):
    if pdf_file is None:
        return ""
    try:
        reader = PyPDF2.PdfReader(pdf_file.name)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        return f"⚠️ Error reading PDF: {str(e)}"

def requirement_analysis(pdf_file, prompt_text):
    content = extract_text_from_pdf(pdf_file) if pdf_file else prompt_text
    analysis_prompt = (
        "Analyze the following requirements and organize them into:\n"
        "- Functional Requirements\n"
        "- Non-Functional Requirements\n"
        "- Technical Specifications\n\n"
        f"{content}"
    )
    return generate_response(analysis_prompt, max_length=400)

def code_generation(prompt, language):
    code_prompt = f"Generate {language} code for the following requirement:\n\n{prompt}\n\nCode:"
    return generate_response(code_prompt, max_length=400)

# ===============================
# 4. Gradio App
# ===============================
with gr.Blocks() as app:
    gr.Markdown("# 🚀 AI SDLC Project: Requirement Analysis & Code Generator")

    with gr.Tabs():
        # Requirements Analysis
        with gr.TabItem("Requirements Analysis"):
            with gr.Row():
                with gr.Column():
                    pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
                    prompt_input = gr.Textbox(
                        label="Or write requirements here",
                        placeholder="Describe your software requirements...",
                        lines=5
                    )
                    analyze_btn = gr.Button("Analyze")
                with gr.Column():
                    analysis_output = gr.Textbox(label="Requirements Analysis", lines=20)
            analyze_btn.click(requirement_analysis, inputs=[pdf_upload, prompt_input], outputs=analysis_output)

        # Code Generation
        with gr.TabItem("Code Generation"):
            with gr.Row():
                with gr.Column():
                    code_prompt = gr.Textbox(
                        label="Code Requirements",
                        placeholder="Describe what code you want to generate...",
                        lines=5
                    )
                    language_dropdown = gr.Dropdown(
                        choices=["Python", "JavaScript", "Java", "C++", "C#", "PHP", "Go", "Rust"],
                        label="Programming Language",
                        value="Python"
                    )
                    generate_btn = gr.Button("Generate Code")
                with gr.Column():
                    code_output = gr.Textbox(label="Generated Code", lines=20)
            generate_btn.click(code_generation, inputs=[code_prompt, language_dropdown], outputs=code_output)

# ===============================
# 5. Launch App
# ===============================
app.launch(share=True)

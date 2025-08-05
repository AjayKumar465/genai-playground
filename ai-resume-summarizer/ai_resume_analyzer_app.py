import gradio as gr
from resume_summarizer import analyze_resume

gr.Interface(
    fn=analyze_resume,
    inputs=[
        gr.File(label="Upload Resume (PDF or DOCX)"),
        gr.Dropdown(choices=["bart", "t5", "pegasus"], value="bart", label="Choose Model"),
        gr.Checkbox(label="Use Structured Prompt for Deep Analysis")  # <-- REMOVE type="value"
    ],
    outputs=gr.Textbox(label="Resume Summary / Analysis"),
    title="🧠 AI Resume Summarizer & Analyzer",
    description="Upload a resume to generate a concise summary or analyze it for key skills, companies, and more."
).launch()

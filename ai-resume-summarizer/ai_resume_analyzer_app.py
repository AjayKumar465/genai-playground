import gradio as gr
from resume_summarizer import process_resume

gr.Interface(
    fn=process_resume,
    inputs=[
        gr.File(label="Upload Resume (PDF or DOCX)"),
        gr.Dropdown(choices=["bart", "t5", "pegasus"], value="bart", label="Choose Model"),
        gr.Slider(30, 200, step=10, label="Min Length", default=30),
        gr.Slider(60, 300, step=10, label="Max Length", default=120),
        gr.Checkbox(label="Use Structured Prompt for Deep Analysis")  # <-- REMOVE type="value"
    ],
   outputs=[
        gr.outputs.Textbox(label="Summary"),
        gr.File(label="Download Summary"),
        gr.Image(label="Skill Frequency Chart")
    ],
    title="AI Resume Summarizer & Analyzer",
    description="Upload a resume to generate a concise summary or analyze it for key skills, companies, and more."
).launch()

import gradio as gr
from summarizer import summarize_text  # uses manual tokenizer + model

def summarize_input(text, model_choice):
    try:
        return summarize_text(text, model_choice)
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

gr.Interface(
    fn=summarize_input,
    inputs=[
        gr.Textbox(lines=10, label="Enter text to summarize"),
        gr.Dropdown(choices=["bart", "t5", "pegasus"], value="bart", label="Choose Model")
    ],
    outputs=gr.Textbox(label="Summary"),
    title="🧠 Text Summarizer",
    description="Select a model (BART, T5, PEGASUS) and enter your input text to generate a summary."
).launch()

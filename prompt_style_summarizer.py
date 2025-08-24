# prompt_style_summarizer.py
from summarizer import summarize_text

# Sample input text
input_text = """
Ajay is a Java developer with 9 years of experience in backend development,
microservices architecture, and system integration. He is currently transitioning into Generative AI, 
working on summarization projects using T5 and BART models with Hugging Face and Gradio.
"""

# Prompt styles
styles = {
    "Basic": f"summarize: {input_text}",
    "Instructional": f"Write a short summary of the following content:\n\n{input_text}",
    "Contextual": f"You are a resume summarizer. Provide a 3-line summary:\n\n{input_text}"
}

# Generate and print summaries
print("\n🔍 Prompt Style Comparisons:\n")
for style, prompt in styles.items():
    summary = summarize_text(prompt, model_choice="t5", max_length=500, min_length=10)
    print(f"--- {style} ---")
    print(summary)
    print()

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_MAP = {
    "bart": "facebook/bart-large-cnn",
    "t5": "t5-small",
    "pegasus": "google/pegasus-xsum"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache loaded models/tokenizers
_loaded_models = {}

def load_model_and_tokenizer(model_choice):
    if model_choice in _loaded_models:
        return _loaded_models[model_choice]

    model_name = MODEL_MAP[model_choice]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    _loaded_models[model_choice] = (tokenizer, model)
    return tokenizer, model

def summarize_text(text, model_choice="bart", max_length=130, min_length=30):
    model_choice = model_choice.lower()
    if model_choice not in MODEL_MAP:
        raise ValueError(f"Unsupported model: {model_choice}")

    tokenizer, model = load_model_and_tokenizer(model_choice)

    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

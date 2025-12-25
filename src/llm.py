"""
LLM Module - TinyLlama inference wrapper with 4-bit quantization support
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_model = None
_tokenizer = None
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def get_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )


def load_model(use_quantization=True):
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    
    print(f"Loading model: {MODEL_NAME}")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _tokenizer.pad_token = _tokenizer.eos_token
    
    if use_quantization and torch.cuda.is_available():
        print("Using 4-bit quantization...")
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=get_quantization_config(),
            device_map="auto",
            trust_remote_code=True
        )
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
    print("Model loaded!")
    return _model, _tokenizer


def format_messages(messages):
    formatted = ""
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "system":
            formatted += f"<|system|>\n{content}</s>\n"
        elif role == "user":
            formatted += f"<|user|>\n{content}</s>\n"
        elif role == "assistant":
            formatted += f"<|assistant|>\n{content}</s>\n"
    formatted += "<|assistant|>\n"
    return formatted


def generate(messages, max_new_tokens=256):
    model, tokenizer = load_model()
    prompt = format_messages(messages)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

"""Edge AI Agent - Futuristic Cyberpunk Web UI"""
import gradio as gr
import re
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None:
        print(f"Loading {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        print("Model loaded!")
    return model, tokenizer


def search_web(query):
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json"
        data = requests.get(url, timeout=10).json()
        if data.get("Abstract"): return data["Abstract"]
        elif data.get("RelatedTopics"):
            results = [t.get("Text", "") for t in data["RelatedTopics"][:3] if t.get("Text")]
            return "\n".join(results) if results else "No results"
        return "No results"
    except Exception as e: return f"Error: {e}"

def get_weather(city):
    try: return requests.get(f"https://wttr.in/{city}?format=3", timeout=10).text.strip()
    except Exception as e: return f"Error: {e}"

TOOLS = {"search": search_web, "weather": get_weather}


def generate(messages, max_tokens=150):
    m, t = load_model()
    prompt = ""
    for msg in messages:
        r, c = msg["role"], msg["content"]
        if r == "system": prompt += f"<|system|>\n{c}</s>\n"
        elif r == "user": prompt += f"<|user|>\n{c}</s>\n"
        elif r == "assistant": prompt += f"<|assistant|>\n{c}</s>\n"
    prompt += "<|assistant|>\n"
    inputs = t(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = m.generate(**inputs, max_new_tokens=max_tokens,
            do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=t.eos_token_id)
    return t.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()


SYSTEM_PROMPT = """You are a helpful AI assistant. You have tools:
- search: Web search (input: query)
- weather: Get weather (input: city name)

If you need a tool, respond EXACTLY like this:
<action>tool_name</action>
<input>tool_input</input>

Otherwise, just respond directly."""

def parse_tool(response):
    action = re.search(r'<action>(.*?)</action>', response, re.DOTALL)
    inp_m = re.search(r'<input>(.*?)</input>', response, re.DOTALL)
    if action and inp_m: return action.group(1).strip(), inp_m.group(1).strip()
    return None, None

def run_agent(message, history):
    try:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": message}]
        for _ in range(3):
            resp = generate(msgs)
            tool, inp = parse_tool(resp)
            if tool and tool in TOOLS:
                result = TOOLS[tool](inp)
                msgs.append({"role": "assistant", "content": resp})
                msgs.append({"role": "user", "content": f"Tool result: {result}"})
            else: return resp
        return resp
    except Exception as e: return f"Error: {str(e)}"


# Futuristic Cyberpunk CSS
STYLES = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&family=Rajdhani:wght@300;400;500&display=swap');

:root {
    --neon-cyan: #00fff2;
    --neon-purple: #bf00ff;
    --neon-pink: #ff0088;
    --dark-bg: #0a0a1a;
    --card-bg: rgba(15, 15, 35, 0.8);
}

/* Dark Background with Grid */
.gradio-container {
    background: 
        linear-gradient(90deg, rgba(0, 255, 242, 0.03) 1px, transparent 1px),
        linear-gradient(rgba(0, 255, 242, 0.03) 1px, transparent 1px),
        radial-gradient(ellipse at 50% 0%, rgba(191, 0, 255, 0.15) 0%, transparent 50%),
        var(--dark-bg) !important;
    background-size: 50px 50px, 50px 50px, 100% 100%, 100% 100% !important;
    max-width: 1000px !important;
    margin: auto;
    min-height: 100vh;
}

/* Header Styles */
.cyber-header {
    text-align: center;
    padding: 40px 20px;
    background: linear-gradient(135deg, rgba(0, 255, 242, 0.1) 0%, rgba(191, 0, 255, 0.1) 100%);
    border-radius: 20px;
    margin-bottom: 25px;
    border: 1px solid rgba(0, 255, 242, 0.3);
    position: relative;
    overflow: hidden;
}

.cyber-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0, 255, 242, 0.2), transparent);
    animation: scan 4s linear infinite;
}

@keyframes scan {
    0% { left: -100%; }
    100% { left: 100%; }
}

.header-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 3em;
    font-weight: 700;
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-purple), var(--neon-pink));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 0 30px rgba(0, 255, 242, 0.5);
    margin-bottom: 10px;
}

.header-sub {
    font-family: 'Rajdhani', sans-serif;
    color: rgba(0, 255, 242, 0.8);
    font-size: 1.2em;
    letter-spacing: 2px;
}

/* Tool Cards */
.tool-card {
    background: var(--card-bg);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 25px;
    border: 1px solid rgba(0, 255, 242, 0.3);
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.tool-card:hover {
    border-color: var(--neon-cyan);
    box-shadow: 0 0 30px rgba(0, 255, 242, 0.3), inset 0 0 30px rgba(0, 255, 242, 0.1);
    transform: translateY(-5px);
}

.tool-icon {
    font-size: 2.5em;
    display: block;
    margin-bottom: 12px;
    filter: drop-shadow(0 0 10px rgba(0, 255, 242, 0.5));
}

.tool-title {
    font-family: 'Orbitron', sans-serif;
    font-weight: 600;
    color: var(--neon-cyan);
    font-size: 1.1em;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.tool-desc {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.9em;
    color: rgba(255, 255, 255, 0.6);
    margin-top: 8px;
}

/* Status Bar */
.status-bar {
    text-align: center;
    padding: 12px;
    background: linear-gradient(90deg, rgba(0, 255, 242, 0.1), rgba(191, 0, 255, 0.1));
    border-radius: 10px;
    margin: 15px 0;
    border: 1px solid rgba(0, 255, 242, 0.2);
}

.status-bar span {
    font-family: 'Rajdhani', sans-serif;
    color: rgba(0, 255, 242, 0.8);
    font-size: 0.95em;
    letter-spacing: 1px;
}

.pulse {
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Chat Interface Styling */
.gradio-container .chatbot {
    background: rgba(10, 10, 26, 0.9) !important;
    border: 1px solid rgba(0, 255, 242, 0.3) !important;
    border-radius: 15px !important;
}

.gradio-container .message {
    font-family: 'Rajdhani', sans-serif;
    letter-spacing: 0.5px;
}

.gradio-container .user {
    background: linear-gradient(135deg, rgba(0, 255, 242, 0.2), rgba(0, 255, 242, 0.1)) !important;
    border: 1px solid rgba(0, 255, 242, 0.3) !important;
}

.gradio-container .bot {
    background: linear-gradient(135deg, rgba(191, 0, 255, 0.2), rgba(191, 0, 255, 0.1)) !important;
    border: 1px solid rgba(191, 0, 255, 0.3) !important;
}

/* Input Box */
.gradio-container textarea, .gradio-container input[type="text"] {
    background: rgba(10, 10, 26, 0.9) !important;
    border: 1px solid rgba(0, 255, 242, 0.3) !important;
    color: #fff !important;
    font-family: 'Rajdhani', sans-serif;
}

.gradio-container textarea:focus, .gradio-container input[type="text"]:focus {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 20px rgba(0, 255, 242, 0.3) !important;
}

/* Buttons */
.gradio-container button.primary {
    background: linear-gradient(135deg, var(--neon-cyan), var(--neon-purple)) !important;
    border: none !important;
    font-family: 'Orbitron', sans-serif;
    text-transform: uppercase;
    letter-spacing: 2px;
    transition: all 0.3s ease;
}

.gradio-container button.primary:hover {
    box-shadow: 0 0 30px rgba(0, 255, 242, 0.5) !important;
    transform: scale(1.02);
}

.gradio-container button.secondary {
    background: transparent !important;
    border: 1px solid rgba(0, 255, 242, 0.5) !important;
    color: var(--neon-cyan) !important;
    font-family: 'Orbitron', sans-serif;
}

/* Footer */
.cyber-footer {
    text-align: center;
    padding: 20px;
    margin-top: 20px;
    border-top: 1px solid rgba(0, 255, 242, 0.2);
}

.cyber-footer a {
    color: var(--neon-cyan);
    text-decoration: none;
    font-family: 'Rajdhani', sans-serif;
    transition: all 0.3s ease;
}

.cyber-footer a:hover {
    text-shadow: 0 0 10px var(--neon-cyan);
}
"""

HEADER_HTML = """
<div class="cyber-header">
    <div class="header-title">▸ EDGE AI AGENT</div>
    <div class="header-sub">LIECU | LIGHTWEIGHT NEURAL AGENT SYSTEM</div>
</div>
"""

TOOL_SEARCH_HTML = """
<div class="tool-card">
    <span class="tool-icon">🔍</span>
    <div class="tool-title">SEARCH</div>
    <div class="tool-desc">Quantum Web Search via DuckDuckGo</div>
</div>
"""

TOOL_WEATHER_HTML = """
<div class="tool-card">
    <span class="tool-icon">🌈</span>
    <div class="tool-title">WEATHER</div>
    <div class="tool-desc">Real-time Atmospheric Data</div>
</div>
"""

STATUS_HTML = """
<div class="status-bar">
    <span class="pulse">◉</span>
    <span>SYSTEM ONLINE | TinyLlama 1.1B | CPU Mode | Response Time: 10-30s</span>
</div>
"""

FOOTER_HTML = """
<div class="cyber-footer">
    <a href="https://github.com/jackeygle/AI_Agent_Project" target="_blank">▸ GITHUB REPOSITORY</a>
    <span style="color: rgba(0, 255, 242, 0.4); margin: 0 15px;">|</span>
    <span style="color: rgba(255, 255, 255, 0.5); font-family: 'Rajdhani', sans-serif;">Built for Edge Deployment Research</span>
</div>
"""

with gr.Blocks(title="Edge AI Agent", css=STYLES, theme=gr.themes.Base()) as demo:
    gr.HTML(HEADER_HTML)
    
    with gr.Row():
        with gr.Column():
            gr.HTML(TOOL_SEARCH_HTML)
        with gr.Column():
            gr.HTML(TOOL_WEATHER_HTML)
    
    gr.HTML(STATUS_HTML)
    
    gr.ChatInterface(
        fn=run_agent,
        examples=[["Hello, initialize system"], ["Weather in Tokyo?"], ["Search: quantum computing"]],
        retry_btn=None, undo_btn=None, clear_btn="⌨ Clear"
    )
    
    gr.HTML(FOOTER_HTML)

if __name__ == "__main__":
    demo.launch()

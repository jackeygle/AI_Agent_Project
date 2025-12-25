# 🤖 Edge AI Agent

A lightweight AI agent demonstrating **tool-calling capabilities** with TinyLlama, designed for edge deployment research.

[![HuggingFace Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/jackeygleee/AI_Agent_Project)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)

## 📖 Overview

This project explores building lightweight AI agents that can run on edge devices. It demonstrates:

- **Tool-calling architecture**: LLM decides when to use external tools
- **Modular design**: Separate modules for LLM, tools, and agent logic
- **Web interface**: Beautiful Gradio-based chat UI

> **Note**: TinyLlama 1.1B is used for demonstration. For reliable tool-calling, larger models (7B+) are recommended.

## 🏗️ Project Structure

```
AI_Agent_Project/
├── app.py              # Gradio web interface (HuggingFace deployment)
├── requirements.txt    # Python dependencies
├── src/
│   ├── __init__.py
│   ├── llm.py          # LLM wrapper with quantization support
│   ├── tools.py        # Tool definitions (search, weather)
│   └── agent.py        # Agent loop with tool parsing
└── notebooks/
    └── 01_getting_started.ipynb  # Jupyter notebook tutorial
```

## 🛠️ Available Tools

| Tool | Description | API |
|------|-------------|-----|
| 🔍 Search | Web search | DuckDuckGo |
| 🌥 Weather | Real-time weather | wttr.in |

## 🚀 Quick Start

### Local Development
```bash
git clone https://github.com/jackeygle/AI_Agent_Project.git
cd AI_Agent_Project
pip install -r requirements.txt
python app.py
```

### HuggingFace Space
Visit: [huggingface.co/spaces/jackeygleee/AI_Agent_Project](https://huggingface.co/spaces/jackeygleee/AI_Agent_Project)

## ⚠️ Limitations

- **Model size**: TinyLlama 1.1B may not reliably follow tool-calling format
- **CPU mode**: Responses take 10-30 seconds
- **Recommended**: Use 7B+ models with GPU for production use

## 📚 Related Topics

- Neural Network Compression
- Agentic AI Efficiency
- Edge Deployment

## 📄 License

MIT License

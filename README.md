# ReVive AI ✨
> From faded to vivid. — Powered by Qubrid AI

## What it does
A 6-agent CrewAI pipeline that restores old, damaged, and black-and-white
photographs using two specialized AI models:
- **Qwen3.5-397B-A17B** (vision) — sees and analyzes the image
- **NVIDIA Nemotron-3-Super-120B-A12B** (reasoning) — plans and executes restoration

## The 6-Agent Crew
1. 🏛️ **Photo Historian** — detects era, context, colorization hints (Qwen vision)
2. 🔍 **Damage Analyst** — maps all damage types and severity (Qwen vision)
3. 📋 **Restoration Strategist** — writes precise PIL/OpenCV repair instructions (Nemotron)
4. 🛠️ **Image Restorer** — executes the restoration brief with PIL/OpenCV tools
5. 🎨 **Colorization Specialist** — adds historically accurate color to B&W photos (Nemotron)
6. ✅ **QA Inspector** — scores result 0-100, triggers retry if score < 60 (Qwen vision)

## Setup

```bash
# 1. Create virtual environment
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
uv sync

# 3. Configure API key
cp .env.example .env
# Edit .env and add your QUBRID_API_KEY

# 4. Run the app
streamlit run app.py
```

## Project structure
```
revive-ai/
├── app.py                      # Main Streamlit entry point
├── crew/
│   ├── agents.py               # All 6 CrewAI agent definitions
│   ├── tasks.py                # Task definitions
│   ├── tools.py                # PIL/OpenCV CrewAI tools
│   └── pipeline.py             # Sequential pipeline orchestration
├── backend/
│   ├── qwen_client.py          # Qwen3.5-397B-A17B API (vision)
│   ├── nemotron_client.py      # Nemotron-120B API (reasoning)
│   └── image_processor.py     # All PIL/OpenCV operations
├── database/
│   └── db.py                   # SQLite jobs & agent logs
├── frontend/
│   ├── components.py           # All UI render functions
│   └── styles.py               # Sepia-to-Vivid CSS theme
├── config/
│   └── settings.py             # Constants, model names, prompts
└── assets/samples/             # 3 synthetic demo photos
```

## Powered by
- [CrewAI](https://crewai.com) — multi-agent orchestration
- [Qwen3.5-397B-A17B](https://qubrid.com) — vision analysis via Qubrid AI
- [NVIDIA Nemotron-3-Super-120B-A12B](https://qubrid.com) — reasoning via Qubrid AI
- PIL + OpenCV — image processing
- SQLite + Streamlit — persistence and UI

# NarrativeFlow

> **My contribution to the NarrativeFlow project** — an Interactive AI Story Co-Writing Platform that lets writers collaborate with AI to craft novels, screenplays, and episodic fiction.

---

## What is NarrativeFlow?

NarrativeFlow is a production-grade narrative engine built with FastAPI. It pairs a writer with an AI co-author that understands context, remembers characters, tracks plotlines, and maintains narrative consistency across long-form stories.

Three writing modes give the writer full control over the creative dynamic:

| Mode | Description |
|------|-------------|
| **AI-Lead** | The AI drives the narrative; the writer steers |
| **User-Lead** | The writer drives; the AI assists and enhances |
| **Co-Author** | Balanced turn-by-turn collaboration |

---

## Features

- **Multi-chapter story management** — create, edit, and organise chapters across full-length works
- **Character & plotline tracking** — dedicated models and routes keep characters and story arcs consistent
- **Story Bible** — a living reference document the AI consults during generation
- **Long-term narrative memory (RAG)** — vector embeddings (pgvector + ChromaDB) let the AI recall earlier events
- **Consistency engine** — flags contradictions in character behaviour, timeline, or world-building
- **AI tools** — recap generation, summarisation, and story-beat suggestions
- **Image prompt generation** — derives scene-accurate prompts for Stable Diffusion (SD-Turbo, DirectML)
- **Audiobook / TTS** — converts chapters to audio via Edge TTS or Kokoro ONNX
- **Export** — DOCX, EPUB, and PDF export powered by `python-docx`, `ebooklib`, and `reportlab`
- **File import** — ingest existing drafts in PDF, DOCX, RTF, TXT, or HTML
- **JWT authentication** — secure user accounts with `python-jose` + `passlib`
- **User AI settings** — per-user API key management and model preferences

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API framework | FastAPI 0.109, Uvicorn |
| Database | PostgreSQL via SQLAlchemy 2 (async) + Alembic migrations |
| Vector store | pgvector, ChromaDB |
| AI / LLM | Google Gemini (`google-generativeai`) |
| Image generation | Diffusers, Stable Diffusion Turbo, ONNX Runtime DirectML |
| TTS | Edge TTS, Kokoro ONNX |
| Auth | JWT (`python-jose`), bcrypt (`passlib`) |
| Export | python-docx, ebooklib, reportlab |
| Import / parsing | PyPDF2, BeautifulSoup4, striprtf, chardet |

---

## Project Structure

```
backend/
├── app/
│   ├── main.py               # FastAPI app, middleware, router registration
│   ├── config.py             # Environment-based settings (pydantic-settings)
│   ├── database.py           # Async SQLAlchemy engine & session factory
│   ├── runtime_settings.py   # Runtime-mutable configuration
│   ├── models/               # SQLAlchemy ORM models
│   │   ├── user.py
│   │   ├── story.py
│   │   ├── chapter.py
│   │   ├── character.py
│   │   ├── plotline.py
│   │   ├── story_bible.py
│   │   ├── embedding.py
│   │   ├── generation.py
│   │   ├── image.py
│   │   ├── user_ai_settings.py
│   │   └── user_api_keys.py
│   ├── routes/               # API endpoint controllers
│   │   ├── auth.py           # /api/auth
│   │   ├── stories.py        # /api/stories
│   │   ├── chapters.py       # /api/chapters
│   │   ├── characters.py     # /api/characters
│   │   ├── plotlines.py      # /api/plotlines
│   │   ├── story_bible.py    # /api/story-bible
│   │   ├── ai_generation.py  # /api/ai
│   │   ├── ai_tools.py       # /api/ai-tools
│   │   ├── memory.py         # /api/memory
│   │   ├── export.py         # /api/export
│   │   ├── images.py         # /api/images
│   │   ├── import_routes.py  # /api/import
│   │   ├── user_settings.py  # /api/settings
│   │   └── audiobook.py      # /api/audiobook
│   └── services/             # Business logic layer
│       ├── gemini_service.py
│       ├── memory_service.py
│       ├── consistency_engine.py
│       ├── story_service.py
│       ├── chapter_service.py
│       ├── character_service.py
│       ├── image_service.py
│       ├── ghibli_image_service.py
│       ├── tts_service.py
│       ├── prompt_builder.py
│       ├── story_extraction.py
│       ├── external_ai_service.py
│       ├── file_parser.py
│       ├── text_utils.py
│       └── token_settings.py
├── add_language_column.py    # DB migration helper
├── add_missing_column.py     # DB migration helper
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL 15+ with the `pgvector` extension
- A Google Gemini API key

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/nikhilvinod321/Narrativeflow-Nikhil_Vinod.git
cd Narrativeflow-Nikhil_Vinod/Narrativeflow/backend

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your database URL and API keys
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `SECRET_KEY` | JWT signing secret |
| `GEMINI_API_KEY` | Google Gemini API key |
| `ENVIRONMENT` | `development` or `production` |

### Running the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs are available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc`.

---

## API Overview

| Prefix | Tag | Purpose |
|--------|-----|---------|
| `/api/auth` | Authentication | Register, login, JWT refresh |
| `/api/stories` | Stories | CRUD for story projects |
| `/api/chapters` | Chapters | Chapter management & ordering |
| `/api/characters` | Characters | Character profiles & arcs |
| `/api/plotlines` | Plotlines | Plot thread tracking |
| `/api/story-bible` | Story Bible | World-building reference |
| `/api/ai` | AI Generation | AI-powered chapter writing |
| `/api/ai-tools` | AI Tools | Recap, summarise, suggest |
| `/api/memory` | Vector Memory | RAG memory CRUD |
| `/api/export` | Export | DOCX / EPUB / PDF download |
| `/api/images` | Image Gallery | Generated image management |
| `/api/audiobook` | Audiobook | TTS audio generation |
| `/api/settings` | User Settings | API keys & model preferences |

---

## Contributor

**Nikhil Vinod** — backend architecture, AI integration, and services layer.

---

## License

This project is part of the NarrativeFlow initiative. See the main repository for licensing details.

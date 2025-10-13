# CubularPyAPI

A high-performance Python API for survey data analysis and AI-powered insights, built with FastAPI, SQLAlchemy, and modern AI providers.

## 🚀 Features

- **FastAPI Framework**: High-performance async web framework with automatic OpenAPI documentation
- **Advanced Authentication**: JWT-based authentication with refresh tokens
- **Survey Management**: Upload, process, and analyze survey data (CSV, Excel formats)
- **Vector Search**: Semantic search across survey responses using embeddings
- **AI Integration**: Chat completions with OpenAI GPT and Anthropic Claude models
- **File Processing**: Automatic data extraction and preprocessing
- **Rate Limiting**: Built-in rate limiting and request logging
- **Type Safety**: Full type hints and Pydantic validation

## 📁 Project Structure

```
CubularPyAPI/
├── app/
│   ├── api/
│   │   ├── endpoints/          # API route handlers
│   │   │   ├── auth.py         # Authentication endpoints
│   │   │   ├── users.py        # User management
│   │   │   ├── surveys.py      # Survey CRUD operations
│   │   │   ├── vector_search.py # Semantic search endpoints
│   │   │   └── chat.py         # AI chat endpoints
│   │   └── api.py              # API router configuration
│   ├── core/
│   │   ├── config.py           # Application configuration
│   │   ├── database.py         # Database connection and session management
│   │   ├── security.py         # Authentication and password hashing
│   │   └── dependencies.py     # FastAPI dependencies
│   ├── middleware/
│   │   ├── loggingMiddleware.py # Request/response logging
│   │   └── rateLimitMiddleware.py # Rate limiting
│   ├── models/
│   │   ├── models.py           # SQLAlchemy database models
│   │   └── schemas.py          # Pydantic request/response schemas
│   ├── services/
│   │   ├── auth_service.py     # Authentication business logic
│   │   ├── survey_service.py   # Survey processing and management
│   │   ├── vector_search_service.py # Semantic search and embeddings
│   │   ├── ai_service.py       # AI chat completions
│   │   └── base.py             # Base service class
│   └── utils/
│       └── logging.py          # Structured logging setup
├── main.py                     # FastAPI application entry point
├── start_server.py             # Server startup script
├── requirements.txt            # Python dependencies
├── .env.template              # Environment configuration template
└── README.md                  # This file
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- PostgreSQL database
- OpenAI API key (optional, for AI features)
- Anthropic API key (optional, for Claude models)

### 1. Environment Setup

```bash
# Clone or navigate to the project directory
cd CubularPyAPI

# Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Configuration

Ensure your PostgreSQL database is running and create a database:

```sql
CREATE DATABASE cubular_db;
CREATE USER cubular_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE cubular_db TO cubular_user;
```

### 3. Environment Variables

Copy the environment template and configure:

```bash
cp .env.template .env
```

Edit `.env` with your settings:

```env
# Required settings
DATABASE_URL=postgresql+asyncpg://cubular_user:your_password@localhost:5432/cubular_db
SECRET_KEY=your-super-secret-key-here-change-in-production

# Optional AI providers
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here
```

### 4. Run the Application

```bash
# Using the startup script
python start_server.py

# Or directly with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🤖 AI Providers

### OpenAI Integration
- **Models**: GPT-4o, GPT-4o-mini, GPT-3.5-turbo
- **Features**: Chat completions, embeddings, streaming responses
- **Configuration**: Set `OPENAI_API_KEY` in environment

### Anthropic Claude Integration
- **Models**: Claude-3-haiku, Claude-3-sonnet, Claude-3-opus
- **Features**: Chat completions, streaming responses
- **Configuration**: Set `ANTHROPIC_API_KEY` in environment

### Local Embeddings Fallback
If no AI provider is configured, the system uses local TF-IDF + SVD embeddings for basic semantic search functionality.

## 📝 Migration from TypeScript API

This Python API is designed to replace the existing TypeScript + Python hybrid system. Key improvements:

- **Unified Codebase**: Single Python codebase instead of dual TS/Python
- **Better Performance**: Native async Python with optimized vector operations
- **Enhanced AI Integration**: Direct integration with multiple AI providers
- **Improved Error Handling**: Comprehensive error tracking and logging
- **Scalable Architecture**: Modern microservice-ready design

---

**CubularPyAPI** - High-performance survey analysis with AI-powered insights
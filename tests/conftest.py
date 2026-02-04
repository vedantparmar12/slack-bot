from __future__ import annotations

import os

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Set test environment before imports
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///test.db"
os.environ["REDIS_URL"] = "redis://localhost:6379"
os.environ["QDRANT_URL"] = "http://localhost:6333"


@pytest_asyncio.fixture
async def client():
    from src.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_markdown():
    return """# Getting Started

## Installation

Run the following command to install:

```bash
pip install our-tool
```

## Configuration

Set the following environment variables:

- `API_KEY`: Your API key
- `API_URL`: The API endpoint URL

### Advanced Configuration

You can also configure via a YAML file:

```yaml
api:
  key: your-key
  url: https://api.example.com
```

## Usage

Import and use the client:

```python
from our_tool import Client
client = Client()
result = client.search("hello")
```
"""


@pytest.fixture
def sample_slack_thread():
    return """[U123]: How do I reset my password?

[U456]: You can reset your password by going to Settings > Security > Change Password.

[U123]: Thanks! What if I forgot my current password?

[U456]: Click "Forgot Password" on the login page. You'll receive a reset link via email. The link expires in 24 hours."""

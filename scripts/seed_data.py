"""Seed the system with sample data for testing and development."""
from __future__ import annotations

import asyncio

from src.ingestion.chunker import SmartChunker
from src.ingestion.embedder import Embedder
from src.ingestion.base import RawDocument

SAMPLE_DOCS = [
    RawDocument(
        source_type="confluence",
        source_id="confluence:SUPPORT:1001",
        title="Password Reset Guide",
        content="""# Password Reset Guide

## How to Reset Your Password

1. Go to the login page
2. Click "Forgot Password"
3. Enter your email address
4. Check your inbox for the reset link
5. Click the link and create a new password

## Password Requirements

- Minimum 12 characters
- At least one uppercase letter
- At least one number
- At least one special character

## Troubleshooting

### I didn't receive the reset email

- Check your spam/junk folder
- Ensure you're using the correct email address
- Wait up to 5 minutes for delivery
- Contact support if the issue persists

### The reset link expired

Reset links expire after 24 hours. Request a new one from the login page.
""",
        url="https://confluence.example.com/display/SUPPORT/Password+Reset+Guide",
        metadata={"space_key": "SUPPORT", "labels": ["password", "authentication"]},
    ),
    RawDocument(
        source_type="slack",
        source_id="slack:C123:1704067200.000",
        title="#support - How do I connect to the VPN?",
        content="""[alice]: How do I connect to the VPN? I'm working from home for the first time.

[bob]: Welcome! Here's how to connect:
1. Download the VPN client from https://vpn.company.com/download
2. Install it and open the application
3. Enter your company email and password
4. Select the "Corporate" profile
5. Click Connect

[alice]: Thanks! It's asking for a 2FA code - where do I get that?

[bob]: You should have the Authenticator app set up. Open it and enter the 6-digit code shown for "Company VPN". If you haven't set up 2FA yet, contact IT at it-help@company.com.

[alice]: Got it working, thanks!""",
        url="https://slack.com/archives/C123/p1704067200000",
        metadata={"channel_name": "support", "participants": ["alice", "bob"]},
    ),
    RawDocument(
        source_type="git",
        source_id="git:dev-portal:docs/api/authentication.md",
        title="API Authentication Guide",
        content="""# API Authentication

## Overview

Our API uses OAuth 2.0 for authentication. All API requests must include a valid access token.

## Getting an Access Token

### Client Credentials Flow

For server-to-server communication:

```bash
curl -X POST https://api.company.com/oauth/token \\
  -d "grant_type=client_credentials" \\
  -d "client_id=YOUR_CLIENT_ID" \\
  -d "client_secret=YOUR_CLIENT_SECRET"
```

### Authorization Code Flow

For user-facing applications:

1. Redirect user to: `https://api.company.com/oauth/authorize?client_id=YOUR_ID&response_type=code`
2. User approves access
3. Exchange the authorization code for a token

## Using the Token

Include the token in the Authorization header:

```
Authorization: Bearer YOUR_ACCESS_TOKEN
```

## Token Expiration

- Access tokens expire after 1 hour
- Use refresh tokens to get new access tokens
- Refresh tokens expire after 30 days

## Rate Limits

- 100 requests per minute per token
- 429 Too Many Requests response when exceeded
- Retry-After header indicates wait time
""",
        url="https://github.com/company/dev-portal/blob/main/docs/api/authentication.md",
        metadata={"repo": "dev-portal", "file_path": "docs/api/authentication.md"},
    ),
]


async def seed():
    from src.db.session import init_db

    await init_db()

    chunker = SmartChunker()
    embedder = Embedder()

    for raw_doc in SAMPLE_DOCS:
        print(f"Seeding: {raw_doc.title}")

        chunks = chunker.chunk(
            content=raw_doc.content,
            source_type=raw_doc.source_type,
            metadata=raw_doc.metadata,
        )

        print(f"  Chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"  [{i}] {chunk.token_count} tokens: {chunk.text[:80]}...")

        # In production, would also embed and upsert to Qdrant
        # point_ids = await embedder.embed_and_upsert(...)

    print(f"\nSeeded {len(SAMPLE_DOCS)} documents")


if __name__ == "__main__":
    asyncio.run(seed())

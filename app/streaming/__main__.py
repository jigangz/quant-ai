from __future__ import annotations

"""Allow running via: python -m app.streaming.service"""
import asyncio
from app.streaming.service import main

asyncio.run(main())

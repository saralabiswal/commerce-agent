"""
Shared pytest fixtures for CommerceAgent tests.
"""
import asyncio
import os
import sys

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Event loop ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    """Single event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Mock LLM provider ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_provider():
    """
    A mock LLMProvider that returns realistic JSON responses
    without making real API calls.
    """
    from llm.base import LLMProvider, LLMResponse

    class MockProvider(LLMProvider):
        def __init__(self):
            self._responses = {}  # keyword → response
            self._call_count = 0
            self.last_messages = []

        @property
        def provider_name(self) -> str:
            return "mock"

        @property
        def model_name(self) -> str:
            return "mock-model"

        def set_response(self, keyword: str, response: str):
            """Set a response to return when keyword appears in the prompt."""
            self._responses[keyword] = response

        async def complete(self, messages, system=None, temperature=0.3, max_tokens=2048):
            self._call_count += 1
            self.last_messages = messages

            # Find matching response
            prompt = " ".join(m.content for m in messages) + (system or "")
            for keyword, response in self._responses.items():
                if keyword.lower() in prompt.lower():
                    return LLMResponse(
                        content=response,
                        model="mock-model",
                        input_tokens=100,
                        output_tokens=200,
                        latency_ms=50.0,
                    )

            # Default response
            return LLMResponse(
                content='{"result": "mock response"}',
                model="mock-model",
                input_tokens=100,
                output_tokens=50,
                latency_ms=10.0,
            )

    return MockProvider()


# ── Sample data ───────────────────────────────────────────────────────────────

@pytest.fixture
def sample_listing():
    return {
        "asin": "DEMO-SKU-001",
        "sku": "DEMO-SKU-001",
        "retailer": "amazon",
        "title": "Wireless Bluetooth Headphones Noise Canceling",
        "bullet_points": [
            "Good sound quality",
            "Long battery life up to 30 hours",
            "Comfortable to wear",
            "Works with phones and computers",
            "Comes with carrying case",
        ],
        "description": "These wireless headphones are great for music and calls.",
        "backend_keywords": "headphones bluetooth wireless",
        "category": "electronics",
        "brand": "SoundWave",
        "price": 79.99,
        "bsr": 1843,
        "review_count": 2341,
        "review_rating": 4.1,
    }


@pytest.fixture
def sample_requirements():
    return {
        "retailer": "amazon",
        "category": "electronics",
        "title": {
            "max_chars": 200,
            "min_chars": 20,
            "prohibited_words": ["best", "#1", "guaranteed"],
        },
        "bullet_points": {
            "count": 5,
            "max_chars_each": 255,
            "start_with_capital": True,
        },
        "description": {
            "max_chars": 2000,
        },
        "backend_keywords": {
            "max_chars": 250,
        },
        "search_volume_benchmarks": {
            "bluetooth headphones": 450000,
            "wireless headphones": 380000,
            "noise cancelling headphones": 290000,
            "over ear headphones": 210000,
            "headphones with microphone": 180000,
        },
    }


@pytest.fixture
def sample_product_specs():
    return {
        "sku": "DEMO-SKU-001",
        "brand": "SoundWave",
        "product_name": "SoundWave Pro X1 Wireless Headphones",
        "product_type": "Over-Ear Wireless Headphones",
        "model_number": "SW-PRX1-BLK",
        "specifications": {
            "driver_size": "40mm dynamic drivers",
            "battery_life": "30 hours playback",
            "connectivity": "Bluetooth 5.2, 3.5mm audio jack",
            "anc": "Active Noise Cancellation with 3 adjustable levels",
            "microphone": "Dual beamforming microphones",
            "weight": "250g",
            "foldable": True,
            "multipoint": "Connect to 2 devices simultaneously",
            "water_resistance": "IPX4 splash resistant",
        },
        "brand_guidelines": {
            "tone": "confident, technical, approachable",
            "prohibited_claims": ["best headphones in the world", "industry-leading"],
            "key_differentiators": ["30-hour battery", "IPX4 rated", "USB-C fast charging"],
            "target_audience": "remote workers, commuters",
        },
    }


@pytest.fixture
def sample_optimized_content():
    return {
        "title": "SoundWave Pro X1 Wireless Bluetooth Headphones — Active Noise Cancellation, 30-Hour Battery, Dual Beamforming Mics, IPX4, Bluetooth 5.2",
        "bullet_points": [
            "IMMERSIVE NOISE CANCELLATION — 3 adjustable ANC levels block distractions in open offices, cafes, and commutes with 40mm dynamic drivers tuned for accurate, detailed sound",
            "ALL-DAY BATTERY, FAST CHARGE — 30 hours of continuous playback on a single charge; 10-minute USB-C charge delivers 3 hours of listening — no more dead headphones mid-meeting",
            "CRYSTAL-CLEAR CALLS ANYWHERE — Dual beamforming microphones with echo cancellation isolate your voice from background noise so every call sounds professional, from home or on the go",
            "BUILT FOR YOUR LIFE — IPX4 splash resistance handles rain and sweat; foldable design packs into the included carrying case; Bluetooth 5.2 connects to 2 devices simultaneously",
            "COMFORTABLE ALL DAY — 250g lightweight build with memory foam protein leather ear cushions; 3.5mm audio cable included for wired listening when battery is low",
        ],
        "description": "The SoundWave Pro X1 wireless headphones are built for the way you actually work and live. Three levels of Active Noise Cancellation adapt to your environment.",
        "backend_keywords": "over ear headphones noise cancelling wireless headphones with microphone foldable headphones bluetooth 5.2 headphones ipx4 headphones",
        "category": "electronics",
    }


@pytest.fixture
def temp_db(tmp_path):
    """Temporary SQLite database for testing."""
    db_path = str(tmp_path / "test_runs.db")
    return f"sqlite+aiosqlite:///{db_path}"


@pytest.fixture
def temp_chroma(tmp_path):
    """Temporary ChromaDB directory for testing."""
    return str(tmp_path / "chroma_test")

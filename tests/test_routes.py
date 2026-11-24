import asyncio
import json
from unittest.mock import MagicMock, patch

from app.api.routes import ask_stream


async def collect(iterator):
    return [item async for item in iterator]


async def events_from(coro):
    response = await coro
    return [json.loads(line[6:]) for line in await collect(response.body_iterator)]


def test_stream_keeps_going_after_empty_first_token():
    def fake_stream(*args, **kwargs):
        yield "", None
        yield "Answer ", None
        yield "[a.txt]", None
        yield "", [("a.txt", None)]

    with patch("app.api.routes.get_vs", return_value=MagicMock()), patch(
        "app.api.routes.stream_answer", side_effect=fake_stream
    ):
        events = asyncio.run(events_from(ask_stream(question="q")))

    assert [e["type"] for e in events] == ["chunk", "chunk", "sources", "done"]
    assert "".join(e["content"] for e in events if e["type"] == "chunk") == "Answer [a.txt]"
    assert events[2]["content"] == [["a.txt", None]]


def test_stream_reports_fallback_when_no_docs():
    def fake_stream(*args, **kwargs):
        yield "", []

    with patch("app.api.routes.get_vs", return_value=MagicMock()), patch(
        "app.api.routes.stream_answer", side_effect=fake_stream
    ):
        events = asyncio.run(events_from(ask_stream(question="q")))

    assert [e["type"] for e in events] == ["answer", "sources", "done"]
    assert events[0]["content"] == "I don't have enough information in the provided documents."

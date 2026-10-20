from datetime import datetime, timedelta, timezone

from fidelitymem.types import Message
from fidelitymem.window import AdaptiveWindow


def _msg(content, **kw):
    return Message(content=content, **kw)


def test_token_pressure_evicts_low_retention_and_emits_boundary():
    fired = []
    w = AdaptiveWindow(min_tokens=10, max_tokens=60, on_boundary=fired.append)
    for i in range(12):
        w.add(_msg(f"filler message number {i} about nothing in particular here"))
    assert w.tokens <= 60
    assert fired and fired[0].reason == "token_pressure"
    assert all(m.content for m in fired[0].messages)


def test_active_task_messages_are_protected():
    w = AdaptiveWindow(min_tokens=40, max_tokens=60)
    w.start_task("t1")
    w.add(_msg("debugging the billing-api timeout, hypothesis is pool exhaustion", task_id="t1", importance=0.9))
    for i in range(8):
        w.add(_msg(f"unrelated chatter {i} about lunch and parking spots"))
    assert any(m.task_id == "t1" for m in w.messages)


def test_topic_shift_and_inactivity_boundaries():
    fired = []
    w = AdaptiveWindow(min_tokens=10, max_tokens=5000, on_boundary=fired.append, inactivity=timedelta(minutes=5))
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(4):
        w.add(_msg(f"billing-api migration {i} failed on PostgreSQL again", timestamp=t0 + timedelta(seconds=i)))
    w.add(_msg("Tomorrow the Zanzibar-trip itinerary needs hotels and flights booked", timestamp=t0 + timedelta(seconds=10)))
    assert "topic_shift" in w.boundaries_fired
    w.add(_msg("flights are booked, hotels pending", timestamp=t0 + timedelta(hours=2)))
    assert "inactivity" in w.boundaries_fired


def test_select_respects_budget_and_prefers_relevance():
    w = AdaptiveWindow(min_tokens=10, max_tokens=5000)
    w.add(_msg("the parking lot is closed on Mondays"))
    w.add(_msg("billing-api pool is capped at 240 connections"))
    w.add(_msg("lunch options include tacos and ramen"))
    sel = w.select("what is the billing-api pool cap", budget_tokens=12)
    assert any("billing-api" in m.content for m in sel)
    assert sum(len(m.content) // 4 for m in sel) <= 12 + 3

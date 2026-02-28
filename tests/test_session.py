"""Unit tests for fastapi_mcp_router.session module.

Tests SessionStore ABC, Session dataclass, and InMemorySessionStore
across all acceptance criteria: AC-16, AC-17, AC-21/AC-89, AC-88, AC-97, AC-98.
"""

import asyncio
import inspect
from datetime import UTC, datetime, timedelta

import pytest

from fastapi_mcp_router.session import InMemorySessionStore, ProgressTracker, Session, SessionStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(ttl_seconds: int = 3600) -> InMemorySessionStore:
    return InMemorySessionStore(ttl_seconds=ttl_seconds)


async def _create_session(store: InMemorySessionStore) -> Session:
    return await store.create(
        protocol_version="2025-06-18",
        client_info={"name": "test-client"},
        capabilities={},
    )


# ---------------------------------------------------------------------------
# AC-16: SessionStore ABC has exactly 6 abstract methods
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_session_store_has_six_abstract_methods():
    """SessionStore ABC exposes exactly 6 abstract methods (AC-16)."""
    abstract_methods = {
        name for name, member in inspect.getmembers(SessionStore) if getattr(member, "__isabstractmethod__", False)
    }
    assert abstract_methods == {
        "create",
        "get",
        "update",
        "delete",
        "enqueue_message",
        "dequeue_messages",
    }


@pytest.mark.unit
def test_session_store_cannot_be_instantiated_directly():
    """SessionStore ABC raises TypeError when instantiated without implementing all methods."""
    with pytest.raises(TypeError):
        SessionStore()


@pytest.mark.unit
def test_all_abstract_methods_are_async():
    """All 6 SessionStore abstract methods are declared as coroutine functions."""
    for name in ("create", "get", "update", "delete", "enqueue_message", "dequeue_messages"):
        method = getattr(SessionStore, name)
        assert asyncio.iscoroutinefunction(method), f"{name} must be async"


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_session_dataclass_defaults():
    """Session dataclass initialises message_queue and subscriptions to empty collections."""
    now = datetime.now(UTC)
    session = Session(
        session_id="abc",
        created_at=now,
        last_activity=now,
        protocol_version="2025-06-18",
        client_info={},
        capabilities={},
    )
    assert session.message_queue == []
    assert session.subscriptions == set()


@pytest.mark.unit
def test_session_dataclass_independent_defaults():
    """Two Session instances do not share the same default list/set objects."""
    now = datetime.now(UTC)
    s1 = Session(
        session_id="x",
        created_at=now,
        last_activity=now,
        protocol_version="2025-06-18",
        client_info={},
        capabilities={},
    )
    s2 = Session(
        session_id="x",
        created_at=now,
        last_activity=now,
        protocol_version="2025-06-18",
        client_info={},
        capabilities={},
    )
    assert s1.message_queue is not s2.message_queue
    assert s1.subscriptions is not s2.subscriptions


# ---------------------------------------------------------------------------
# AC-17: InMemorySessionStore with configurable TTL
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_in_memory_store_default_ttl():
    """InMemorySessionStore defaults to ttl_seconds=3600 (AC-17)."""
    store = InMemorySessionStore()
    assert store.ttl_seconds == 3600


@pytest.mark.unit
def test_in_memory_store_custom_ttl():
    """InMemorySessionStore accepts a custom ttl_seconds value (AC-17)."""
    store = InMemorySessionStore(ttl_seconds=120)
    assert store.ttl_seconds == 120


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_returns_session_with_uuid4():
    """create() returns a Session with a non-empty session_id."""
    store = _make_store()
    session = await _create_session(store)
    assert len(session.session_id) == 36  # UUID4 canonical form
    assert session.session_id.count("-") == 4


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_sets_utc_timestamps():
    """create() sets created_at and last_activity to UTC datetimes."""
    before = datetime.now(UTC)
    store = _make_store()
    session = await _create_session(store)
    after = datetime.now(UTC)
    assert before <= session.created_at <= after
    assert before <= session.last_activity <= after
    assert session.created_at.tzinfo is UTC


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_stores_two_unique_sessions():
    """create() produces unique session_ids for successive calls."""
    store = _make_store()
    s1 = await _create_session(store)
    s2 = await _create_session(store)
    assert s1.session_id != s2.session_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_returns_stored_session():
    """get() returns the Session created by create()."""
    store = _make_store()
    session = await _create_session(store)
    found = await store.get(session.session_id)
    assert found is not None
    assert found.session_id == session.session_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_returns_none_for_missing_session():
    """get() returns None for a session_id that was never created."""
    store = _make_store()
    result = await store.get("nonexistent-id")
    assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_updates_last_activity():
    """get() updates last_activity on the session."""
    store = _make_store()
    session = await _create_session(store)
    original_activity = session.last_activity
    await asyncio.sleep(0.01)
    found = await store.get(session.session_id)
    assert found is not None
    assert found.last_activity >= original_activity


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_persists_field_change():
    """update() persists mutations to an existing session."""
    store = _make_store()
    session = await _create_session(store)
    session.protocol_version = "2025-03-26"
    await store.update(session)
    found = await store.get(session.session_id)
    assert found is not None
    assert found.protocol_version == "2025-03-26"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_removes_session():
    """delete() removes the session so subsequent get() returns None."""
    store = _make_store()
    session = await _create_session(store)
    await store.delete(session.session_id)
    assert await store.get(session.session_id) is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_nonexistent_session_is_silent():
    """delete() on an unknown session_id does not raise."""
    store = _make_store()
    await store.delete("does-not-exist")  # must not raise


# ---------------------------------------------------------------------------
# AC-88: dequeue_messages() returns [] for empty queue
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dequeue_empty_queue_returns_empty_list(caplog):
    """dequeue_messages() returns [] when the session queue is empty (AC-88)."""
    store = _make_store()
    session = await _create_session(store)
    result = await store.dequeue_messages(session.session_id)
    assert result == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dequeue_missing_session_returns_empty_list():
    """dequeue_messages() returns [] for an unknown session_id (AC-88)."""
    store = _make_store()
    result = await store.dequeue_messages("no-such-session")
    assert result == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dequeue_returns_and_clears_atomically():
    """dequeue_messages() returns enqueued messages and leaves queue empty."""
    store = _make_store()
    session = await _create_session(store)
    msg1: dict = {"method": "ping"}
    msg2: dict = {"method": "pong"}
    await store.enqueue_message(session.session_id, msg1)
    await store.enqueue_message(session.session_id, msg2)
    messages = await store.dequeue_messages(session.session_id)
    assert messages == [msg1, msg2]
    # Queue must be cleared after dequeue
    second_dequeue = await store.dequeue_messages(session.session_id)
    assert second_dequeue == []


# ---------------------------------------------------------------------------
# AC-21 / AC-89: message queue bounded at 1000
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enqueue_up_to_limit_succeeds():
    """enqueue_message() accepts exactly 1000 messages (AC-21/AC-89)."""
    store = _make_store()
    session = await _create_session(store)
    for i in range(1000):
        await store.enqueue_message(session.session_id, {"index": i})
    found = await store.get(session.session_id)
    assert found is not None
    assert len(found.message_queue) == 1000


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enqueue_1001st_message_is_dropped():
    """The 1001st enqueue_message() call is silently dropped (AC-21/AC-89)."""
    store = _make_store()
    session = await _create_session(store)
    for i in range(1000):
        await store.enqueue_message(session.session_id, {"index": i})
    # 1001st message — must be silently dropped
    await store.enqueue_message(session.session_id, {"index": 1000})
    found = await store.get(session.session_id)
    assert found is not None
    assert len(found.message_queue) == 1000
    # Original 1000 messages are intact; last entry has index 999
    assert found.message_queue[-1] == {"index": 999}


# ---------------------------------------------------------------------------
# IR-12: ProgressTracker — report_progress, request_cancellation,
#         is_cancelled, clear_cancellation; EC-18: exception swallowed
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_report_progress_enqueues_notification_with_message():
    """report_progress() enqueues a notifications/progress dict with message (IR-12)."""
    store = InMemorySessionStore()
    session = await store.create("2025-06-18", {}, {})
    tracker = ProgressTracker(session_store=store)
    await tracker.report_progress(session.session_id, "req-1", 3, 10, "working")
    messages = await store.dequeue_messages(session.session_id)
    assert len(messages) == 1
    assert messages[0] == {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {"progressToken": "req-1", "progress": 3, "total": 10, "message": "working"},
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_report_progress_omits_message_key_when_none():
    """report_progress() omits the message key when message=None (IR-12)."""
    store = InMemorySessionStore()
    session = await store.create("2025-06-18", {}, {})
    tracker = ProgressTracker(session_store=store)
    await tracker.report_progress(session.session_id, "req-1", 5, 10, None)
    messages = await store.dequeue_messages(session.session_id)
    assert len(messages) == 1
    assert "message" not in messages[0]["params"]


@pytest.mark.unit
def test_request_cancellation_and_is_cancelled():
    """request_cancellation() marks a request; is_cancelled() returns True (IR-12)."""
    store = InMemorySessionStore()
    tracker = ProgressTracker(session_store=store)
    assert not tracker.is_cancelled("req-1")
    tracker.request_cancellation("req-1")
    assert tracker.is_cancelled("req-1")


@pytest.mark.unit
def test_clear_cancellation_is_idempotent():
    """clear_cancellation() removes the ID; a second call does not raise (IR-12)."""
    store = InMemorySessionStore()
    tracker = ProgressTracker(session_store=store)
    tracker.request_cancellation("req-1")
    tracker.clear_cancellation("req-1")
    assert not tracker.is_cancelled("req-1")
    tracker.clear_cancellation("req-1")  # second call must not raise


@pytest.mark.unit
@pytest.mark.asyncio
async def test_report_progress_catches_exception_and_continues(caplog):
    """EC-18: exception from enqueue_message is swallowed and logged; no raise."""

    class FailingStore(InMemorySessionStore):
        async def enqueue_message(self, session_id: str, message: dict) -> None:
            raise RuntimeError("boom")

    tracker = ProgressTracker(session_store=FailingStore())
    # Must not raise
    await tracker.report_progress("sess", "req", 1, 10, None)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enqueue_missing_session_is_silent():
    """enqueue_message() on an unknown session_id does not raise."""
    store = _make_store()
    await store.enqueue_message("missing-id", {"method": "ping"})  # must not raise


# ---------------------------------------------------------------------------
# AC-97: get() returns None for expired session
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_returns_none_for_expired_session():
    """get() returns None and deletes a session whose TTL has elapsed (AC-97)."""
    store = _make_store(ttl_seconds=1)
    session = await _create_session(store)
    # Backdate last_activity beyond TTL
    session.last_activity = datetime.now(UTC) - timedelta(seconds=2)
    await store.update(session)
    result = await store.get(session.session_id)
    assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_expired_session_removed_from_store():
    """After get() returns None for an expired session, it is deleted from storage."""
    store = _make_store(ttl_seconds=1)
    session = await _create_session(store)
    session.last_activity = datetime.now(UTC) - timedelta(seconds=2)
    await store.update(session)
    await store.get(session.session_id)
    # Direct internal check — expired session must be gone
    assert session.session_id not in store._sessions


@pytest.mark.unit
@pytest.mark.asyncio
async def test_session_at_ttl_boundary_is_not_expired():
    """A session whose last_activity is exactly at the TTL boundary is still valid."""
    store = _make_store(ttl_seconds=60)
    session = await _create_session(store)
    # 59 seconds old — still within TTL
    session.last_activity = datetime.now(UTC) - timedelta(seconds=59)
    await store.update(session)
    result = await store.get(session.session_id)
    assert result is not None


# ---------------------------------------------------------------------------
# AC-98: concurrent enqueue_message() — asyncio.Lock prevents race
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_concurrent_enqueue_no_race_condition():
    """Concurrent enqueue_message() calls produce no lost writes (AC-98)."""
    store = _make_store()
    session = await _create_session(store)

    async def enqueue_batch(start: int, count: int) -> None:
        for i in range(start, start + count):
            await store.enqueue_message(session.session_id, {"index": i})

    await asyncio.gather(
        enqueue_batch(0, 100),
        enqueue_batch(100, 100),
        enqueue_batch(200, 100),
    )

    found = await store.get(session.session_id)
    assert found is not None
    assert len(found.message_queue) == 300


@pytest.mark.unit
@pytest.mark.asyncio
async def test_concurrent_enqueue_at_capacity_drops_excess():
    """Concurrent enqueue calls beyond 1000 silently drop excess messages."""
    store = _make_store()
    session = await _create_session(store)

    # Fill queue to 900 sequentially to ensure a predictable baseline
    for i in range(900):
        await store.enqueue_message(session.session_id, {"index": i})

    # Concurrently attempt to add 200 more (only 100 fit)
    async def enqueue_batch(start: int, count: int) -> None:
        for i in range(start, start + count):
            await store.enqueue_message(session.session_id, {"index": i})

    await asyncio.gather(
        enqueue_batch(900, 100),
        enqueue_batch(1000, 100),
    )

    found = await store.get(session.session_id)
    assert found is not None
    assert len(found.message_queue) == 1000

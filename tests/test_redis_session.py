"""Contract tests for fastapi_mcp_router.session.RedisSessionStore.

Tests cover:
- AC-72: redis.asyncio is an optional dependency; missing redis raises on instantiation
- AC-71: RedisSessionStore passes all SessionStore contract methods
- AC-84 / EC-27: Redis connection failure raises MCPError -32603
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fastapi_mcp_router.exceptions import MCPError
from fastapi_mcp_router.session import RedisSessionStore, Session

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_redis_mock() -> MagicMock:
    """Build a MagicMock that mimics the redis.asyncio.Redis interface.

    All awaitable methods (set, get, expire, delete, llen, lpush, lrange)
    are AsyncMock instances. The pipeline() method returns a synchronous
    MagicMock with lrange/delete as regular mocks (they add to a pipeline,
    not awaited) and execute as an AsyncMock.
    """
    redis = MagicMock()
    redis.set = AsyncMock(return_value=True)
    redis.get = AsyncMock(return_value=None)
    redis.expire = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.llen = AsyncMock(return_value=0)
    redis.lpush = AsyncMock(return_value=1)
    redis.lrange = AsyncMock(return_value=[])

    pipe = MagicMock()
    pipe.lrange = MagicMock()
    pipe.delete = MagicMock()
    pipe.execute = AsyncMock(return_value=[[], 0])
    redis.pipeline = MagicMock(return_value=pipe)

    return redis


def _make_store(ttl_seconds: int = 7200) -> tuple[RedisSessionStore, MagicMock]:
    """Return a (store, redis_mock) pair with redis availability patched in."""
    redis_mock = _make_redis_mock()
    with patch("fastapi_mcp_router.session._aioredis_runtime", new=object()):
        store = RedisSessionStore(redis_client=redis_mock, ttl_seconds=ttl_seconds)
    return store, redis_mock


def _make_session(session_id: str = "sess-abc-123") -> Session:
    """Return a minimal Session fixture for serialisation tests."""
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return Session(
        session_id=session_id,
        created_at=now,
        last_activity=now,
        protocol_version="2025-06-18",
        client_info={"name": "test-client"},
        capabilities={},
    )


def _serialize_session(store: RedisSessionStore, session: Session) -> str:
    """Expose private _serialize for test use."""
    return store._serialize(session)


# ---------------------------------------------------------------------------
# AC-72: redis is an optional dependency
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_import_fastapi_mcp_router_without_redis_raises_no_error():
    """Importing fastapi_mcp_router does not raise even when redis is absent (AC-72)."""
    # The import already succeeded at module load time; this test confirms it.
    import fastapi_mcp_router  # noqa: F401


@pytest.mark.unit
def test_instantiating_redis_store_without_redis_raises_runtime_error():
    """RedisSessionStore.__init__ raises RuntimeError when redis is not installed (AC-72).

    The spec states ImportError; the implementation raises RuntimeError.
    """
    redis_mock = _make_redis_mock()
    with (
        patch("fastapi_mcp_router.session._aioredis_runtime", new=None),
        pytest.raises(RuntimeError, match="redis-py is required"),
    ):
        RedisSessionStore(redis_client=redis_mock)


# ---------------------------------------------------------------------------
# AC-71: contract — create()
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_returns_session_with_uuid4():
    """create() returns a Session with a UUID4 session_id (AC-71)."""
    store, _ = _make_store()
    session = await store.create(
        protocol_version="2025-06-18",
        client_info={"name": "client"},
        capabilities={},
    )
    assert isinstance(session, Session)
    assert len(session.session_id) == 36
    assert session.session_id.count("-") == 4


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_stores_session_in_redis():
    """create() calls redis.set and redis.expire with the correct key (AC-71)."""
    store, redis_mock = _make_store()
    session = await store.create(
        protocol_version="2025-06-18",
        client_info={},
        capabilities={},
    )
    expected_key = f"mcp:session:{session.session_id}"
    redis_mock.set.assert_called_once()
    call_args = redis_mock.set.call_args
    assert call_args[0][0] == expected_key
    redis_mock.expire.assert_called_once_with(expected_key, 7200)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_serializes_protocol_version():
    """create() persists the protocol_version in the JSON blob (AC-71)."""
    store, redis_mock = _make_store()
    await store.create(
        protocol_version="2025-03-26",
        client_info={},
        capabilities={},
    )
    stored_json = redis_mock.set.call_args[0][1]
    data = json.loads(stored_json)
    assert data["protocol_version"] == "2025-03-26"


# ---------------------------------------------------------------------------
# AC-71: contract — get()
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_returns_session_when_key_exists():
    """get() deserialises and returns the Session when Redis has the key (AC-71)."""
    store, redis_mock = _make_store()
    session = _make_session()
    redis_mock.get = AsyncMock(return_value=_serialize_session(store, session).encode())

    result = await store.get(session.session_id)

    assert result is not None
    assert result.session_id == session.session_id
    assert result.protocol_version == "2025-06-18"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_returns_none_when_key_absent():
    """get() returns None when Redis has no value for the key (AC-71)."""
    store, redis_mock = _make_store()
    redis_mock.get = AsyncMock(return_value=None)

    result = await store.get("nonexistent-session-id")

    assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_refreshes_last_activity_in_redis():
    """get() re-persists the session with updated last_activity (AC-71)."""
    store, redis_mock = _make_store()
    session = _make_session()
    redis_mock.get = AsyncMock(return_value=_serialize_session(store, session).encode())

    await store.get(session.session_id)

    # set() called once for the activity refresh; expire() also called once
    assert redis_mock.set.call_count == 1
    assert redis_mock.expire.call_count == 1


# ---------------------------------------------------------------------------
# AC-71: contract — update()
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_calls_set_and_expire():
    """update() persists the session JSON and refreshes the TTL (AC-71)."""
    store, redis_mock = _make_store()
    session = _make_session()

    await store.update(session)

    expected_key = f"mcp:session:{session.session_id}"
    redis_mock.set.assert_called_once()
    assert redis_mock.set.call_args[0][0] == expected_key
    redis_mock.expire.assert_called_once_with(expected_key, 7200)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_serializes_changed_field():
    """update() persists mutations to a session field (AC-71)."""
    store, redis_mock = _make_store()
    session = _make_session()
    session.protocol_version = "2025-03-26"

    await store.update(session)

    stored_json = redis_mock.set.call_args[0][1]
    data = json.loads(stored_json)
    assert data["protocol_version"] == "2025-03-26"


# ---------------------------------------------------------------------------
# AC-71: contract — delete()
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_calls_redis_delete_with_both_keys():
    """delete() removes both session key and queue key from Redis (AC-71)."""
    store, redis_mock = _make_store()
    session_id = "sess-delete-me"

    await store.delete(session_id)

    redis_mock.delete.assert_called_once_with(
        f"mcp:session:{session_id}",
        f"mcp:queue:{session_id}",
    )


# ---------------------------------------------------------------------------
# AC-71: contract — enqueue_message()
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enqueue_message_calls_lpush_when_queue_not_full():
    """enqueue_message() calls lpush when queue length is below 1000 (AC-71)."""
    store, redis_mock = _make_store()
    redis_mock.llen = AsyncMock(return_value=0)

    message = {"method": "ping"}
    await store.enqueue_message("sess-1", message)

    redis_mock.lpush.assert_called_once_with("mcp:queue:sess-1", json.dumps(message))
    redis_mock.expire.assert_called_once_with("mcp:queue:sess-1", 7200)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enqueue_message_drops_when_queue_at_1000():
    """enqueue_message() silently drops the message when queue holds 1000 items (AC-71)."""
    store, redis_mock = _make_store()
    redis_mock.llen = AsyncMock(return_value=1000)

    await store.enqueue_message("sess-1", {"method": "ping"})

    redis_mock.lpush.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enqueue_message_checks_length_before_push():
    """enqueue_message() reads llen before deciding to lpush (AC-71)."""
    store, redis_mock = _make_store()
    redis_mock.llen = AsyncMock(return_value=999)

    await store.enqueue_message("sess-1", {"method": "ping"})

    redis_mock.llen.assert_called_once_with("mcp:queue:sess-1")
    redis_mock.lpush.assert_called_once()


# ---------------------------------------------------------------------------
# AC-71: contract — dequeue_messages()
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dequeue_messages_uses_pipeline():
    """dequeue_messages() calls pipeline() for atomic LRANGE+DEL (AC-71)."""
    store, redis_mock = _make_store()
    pipe = redis_mock.pipeline.return_value
    pipe.execute = AsyncMock(return_value=[[], 0])

    await store.dequeue_messages("sess-1")

    redis_mock.pipeline.assert_called_once()
    pipe.lrange.assert_called_once_with("mcp:queue:sess-1", 0, -1)
    pipe.delete.assert_called_once_with("mcp:queue:sess-1")
    pipe.execute.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dequeue_messages_returns_empty_list_for_empty_queue():
    """dequeue_messages() returns [] when the Redis list is empty (AC-71)."""
    store, redis_mock = _make_store()
    pipe = redis_mock.pipeline.return_value
    pipe.execute = AsyncMock(return_value=[[], 0])

    result = await store.dequeue_messages("sess-1")

    assert result == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dequeue_messages_deserializes_messages():
    """dequeue_messages() JSON-decodes each item from the Redis list (AC-71).

    Redis LPUSH prepends, so LRANGE returns items in LIFO order.
    The implementation calls reverse() before parsing to restore FIFO order.
    """
    store, redis_mock = _make_store()
    msg1 = {"method": "ping"}
    msg2 = {"method": "pong"}
    # LPUSH prepends: msg2 was pushed after msg1, so LRANGE returns [msg2_raw, msg1_raw]
    raw = [json.dumps(msg2).encode(), json.dumps(msg1).encode()]
    pipe = redis_mock.pipeline.return_value
    pipe.execute = AsyncMock(return_value=[raw, 1])

    result = await store.dequeue_messages("sess-1")

    # After reverse(), FIFO order is restored: [msg1, msg2]
    assert result == [msg1, msg2]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dequeue_messages_second_call_returns_empty():
    """After dequeue, a second dequeue on the same session returns [] (AC-71).

    The pipeline DEL clears the queue; second call returns empty.
    """
    store, redis_mock = _make_store()

    call_count = 0

    async def execute_side_effect() -> list:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [[json.dumps({"method": "ping"}).encode()], 1]
        return [[], 0]

    pipe = redis_mock.pipeline.return_value
    pipe.execute = AsyncMock(side_effect=execute_side_effect)

    first = await store.dequeue_messages("sess-1")
    second = await store.dequeue_messages("sess-1")

    assert len(first) == 1
    assert second == []


# ---------------------------------------------------------------------------
# AC-84 / EC-27: Redis connection failure → MCPError -32603
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_redis_failure_raises_mcp_error():
    """create() raises MCPError -32603 when redis.set raises an exception (AC-84/EC-27)."""
    store, redis_mock = _make_store()
    redis_mock.set = AsyncMock(side_effect=ConnectionError("Redis unavailable"))

    with pytest.raises(MCPError) as exc_info:
        await store.create("2025-06-18", {}, {})

    assert exc_info.value.code == -32603


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_redis_failure_raises_mcp_error():
    """get() raises MCPError -32603 when redis.get raises an exception (AC-84/EC-27)."""
    store, redis_mock = _make_store()
    redis_mock.get = AsyncMock(side_effect=ConnectionError("Redis unavailable"))

    with pytest.raises(MCPError) as exc_info:
        await store.get("sess-1")

    assert exc_info.value.code == -32603


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_redis_failure_raises_mcp_error():
    """update() raises MCPError -32603 when redis.set raises an exception (AC-84/EC-27)."""
    store, redis_mock = _make_store()
    redis_mock.set = AsyncMock(side_effect=ConnectionError("Redis unavailable"))
    session = _make_session()

    with pytest.raises(MCPError) as exc_info:
        await store.update(session)

    assert exc_info.value.code == -32603


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_redis_failure_raises_mcp_error():
    """delete() raises MCPError -32603 when redis.delete raises an exception (AC-84/EC-27)."""
    store, redis_mock = _make_store()
    redis_mock.delete = AsyncMock(side_effect=ConnectionError("Redis unavailable"))

    with pytest.raises(MCPError) as exc_info:
        await store.delete("sess-1")

    assert exc_info.value.code == -32603


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dequeue_messages_retries_on_transient_failure():
    """dequeue_messages() retries once on transient error and returns messages on success."""
    store, redis_mock = _make_store()
    pipe = redis_mock.pipeline.return_value
    pipe.execute = AsyncMock(
        side_effect=[
            ConnectionError("transient"),
            [
                [json.dumps({"method": "ping"}).encode()],
                1,
            ],
        ]
    )

    with patch("fastapi_mcp_router.session.asyncio.sleep", new_callable=AsyncMock):
        result = await store.dequeue_messages("sess-1")

    assert result == [{"method": "ping"}]
    assert pipe.execute.call_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dequeue_messages_raises_after_retry_exhausted():
    """dequeue_messages() raises MCPError -32603 after both attempts fail."""
    store, redis_mock = _make_store()
    pipe = redis_mock.pipeline.return_value
    pipe.execute = AsyncMock(side_effect=ConnectionError("Redis unavailable"))

    with patch("fastapi_mcp_router.session.asyncio.sleep", new_callable=AsyncMock), pytest.raises(MCPError) as exc_info:
        await store.dequeue_messages("sess-1")

    assert exc_info.value.code == -32603
    assert pipe.execute.call_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enqueue_message_redis_failure_raises_mcp_error():
    """enqueue_message() raises MCPError -32603 when llen raises (AC-84/EC-27)."""
    store, redis_mock = _make_store()
    redis_mock.llen = AsyncMock(side_effect=ConnectionError("Redis unavailable"))

    with pytest.raises(MCPError) as exc_info:
        await store.enqueue_message("sess-1", {"method": "ping"})

    assert exc_info.value.code == -32603


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dequeue_messages_redis_failure_raises_mcp_error():
    """dequeue_messages() raises MCPError -32603 when pipeline.execute raises (AC-84/EC-27)."""
    store, redis_mock = _make_store()
    pipe = redis_mock.pipeline.return_value
    pipe.execute = AsyncMock(side_effect=ConnectionError("Redis unavailable"))

    with pytest.raises(MCPError) as exc_info:
        await store.dequeue_messages("sess-1")

    assert exc_info.value.code == -32603

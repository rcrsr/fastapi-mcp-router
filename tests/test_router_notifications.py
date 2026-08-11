"""Tests for MCPRouter notify_* trigger methods.

Verifies list_changed notifications broadcast to all live sessions,
resource_updated notifications fan out only to subscribers, and that
notify_* is a silent no-op when no session_store is configured.

Also verifies the wire method-name strings stay snake_case (while
initialize capability sub-fields stay camelCase), the message queue's
1000-message cap silently drops overflow, and that a notification
reaches an active SSE stream within one dequeue cycle.
"""

import asyncio
import contextlib
import time

import httpx
import pytest
from fastapi import FastAPI

from fastapi_mcp_router import InMemorySessionStore, MCPRouter
from tests.conftest import SseCapture


@pytest.mark.unit
@pytest.mark.asyncio
async def test_notify_tools_list_changed_enqueues_for_all_sessions() -> None:
    """notify_tools_list_changed enqueues the snake_case method for every live session."""
    store = InMemorySessionStore()
    mcp = MCPRouter(session_store=store, stateful=True)
    session_a = await store.create(protocol_version="2025-06-18", client_info={}, capabilities={})
    session_b = await store.create(protocol_version="2025-06-18", client_info={}, capabilities={})

    await mcp.notify_tools_list_changed()

    for session in (session_a, session_b):
        messages = await store.dequeue_messages(session.session_id)
        assert messages == [{"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_notify_resources_and_prompts_list_changed_use_snake_case_methods() -> None:
    """notify_resources_list_changed and notify_prompts_list_changed enqueue snake_case methods."""
    store = InMemorySessionStore()
    mcp = MCPRouter(session_store=store, stateful=True)
    session = await store.create(protocol_version="2025-06-18", client_info={}, capabilities={})

    await mcp.notify_resources_list_changed()
    await mcp.notify_prompts_list_changed()

    messages = await store.dequeue_messages(session.session_id)
    assert messages == [
        {"jsonrpc": "2.0", "method": "notifications/resources/list_changed"},
        {"jsonrpc": "2.0", "method": "notifications/prompts/list_changed"},
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_notify_resource_updated_only_reaches_subscribers() -> None:
    """notify_resource_updated(uri) enqueues notifications/resources/updated only to subscribers."""
    store = InMemorySessionStore()
    mcp = MCPRouter(session_store=store, stateful=True)
    subscriber = await store.create(protocol_version="2025-06-18", client_info={}, capabilities={})
    non_subscriber = await store.create(protocol_version="2025-06-18", client_info={}, capabilities={})
    subscriber.subscriptions.add("file:///doc.txt")
    await store.update(subscriber)

    await mcp.notify_resource_updated("file:///doc.txt")

    subscriber_messages = await store.dequeue_messages(subscriber.session_id)
    assert subscriber_messages == [
        {
            "jsonrpc": "2.0",
            "method": "notifications/resources/updated",
            "params": {"uri": "file:///doc.txt"},
        }
    ]
    non_subscriber_messages = await store.dequeue_messages(non_subscriber.session_id)
    assert non_subscriber_messages == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_notify_methods_are_silent_no_op_without_session_store() -> None:
    """notify_* raises no exception and enqueues nothing when no session_store is configured."""
    mcp = MCPRouter()

    await mcp.notify_tools_list_changed()
    await mcp.notify_resources_list_changed()
    await mcp.notify_prompts_list_changed()
    await mcp.notify_resource_updated("file:///doc.txt")


class _FlakyEnqueueSessionStore(InMemorySessionStore):
    """InMemorySessionStore whose enqueue_message fails for one session id."""

    def __init__(self, failing_session_id: str) -> None:
        super().__init__()
        self._failing_session_id = failing_session_id

    async def enqueue_message(self, session_id: str, message: dict) -> None:
        if session_id == self._failing_session_id:
            raise RuntimeError("simulated transient Redis failure")
        await super().enqueue_message(session_id, message)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_broadcast_list_changed_continues_after_one_session_enqueue_fails() -> None:
    """A single session's enqueue failure does not abort delivery to remaining sessions."""
    store = _FlakyEnqueueSessionStore(failing_session_id="")
    session_a = await store.create(protocol_version="2025-06-18", client_info={}, capabilities={})
    store._failing_session_id = session_a.session_id
    session_b = await store.create(protocol_version="2025-06-18", client_info={}, capabilities={})
    mcp = MCPRouter(session_store=store, stateful=True)

    await mcp.notify_tools_list_changed()

    messages_b = await store.dequeue_messages(session_b.session_id)
    assert messages_b == [{"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}]


class _ConcurrencyTrackingSessionStore(InMemorySessionStore):
    """InMemorySessionStore that records the peak number of in-flight enqueue_message calls."""

    def __init__(self) -> None:
        super().__init__()
        self._in_flight = 0
        self.peak_in_flight = 0

    async def enqueue_message(self, session_id: str, message: dict) -> None:
        self._in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            await asyncio.sleep(0)  # yield control so overlapping calls can interleave
            await super().enqueue_message(session_id, message)
        finally:
            self._in_flight -= 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_broadcast_list_changed_fans_out_concurrently_to_all_sessions() -> None:
    """_broadcast_list_changed enqueues to every session concurrently, not strictly sequentially."""
    store = _ConcurrencyTrackingSessionStore()
    mcp = MCPRouter(session_store=store, stateful=True)
    sessions = [await store.create(protocol_version="2025-06-18", client_info={}, capabilities={}) for _ in range(10)]

    await mcp.notify_tools_list_changed()

    assert store.peak_in_flight > 1, "expected concurrent fan-out, but calls ran strictly sequentially"
    for session in sessions:
        messages = await store.dequeue_messages(session.session_id)
        assert messages == [{"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_notify_resource_updated_fans_out_concurrently_to_all_subscribers() -> None:
    """notify_resource_updated enqueues to every subscriber concurrently, not strictly sequentially."""
    store = _ConcurrencyTrackingSessionStore()
    mcp = MCPRouter(session_store=store, stateful=True)
    subscribers = []
    for _ in range(10):
        session = await store.create(protocol_version="2025-06-18", client_info={}, capabilities={})
        session.subscriptions.add("file:///doc.txt")
        await store.update(session)
        subscribers.append(session)

    await mcp.notify_resource_updated("file:///doc.txt")

    assert store.peak_in_flight > 1, "expected concurrent fan-out, but calls ran strictly sequentially"
    for session in subscribers:
        messages = await store.dequeue_messages(session.session_id)
        assert len(messages) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_broadcast_list_changed_bounds_concurrency_by_semaphore() -> None:
    """Fan-out concurrency never exceeds the configured semaphore bound, even with many sessions."""
    from fastapi_mcp_router.router import _MAX_CONCURRENT_ENQUEUES

    store = _ConcurrencyTrackingSessionStore()
    mcp = MCPRouter(session_store=store, stateful=True)
    session_count = _MAX_CONCURRENT_ENQUEUES + 20
    for _ in range(session_count):
        await store.create(protocol_version="2025-06-18", client_info={}, capabilities={})

    await mcp.notify_tools_list_changed()

    assert store.peak_in_flight <= _MAX_CONCURRENT_ENQUEUES


@pytest.mark.unit
@pytest.mark.asyncio
async def test_notify_drops_silently_at_thousand_message_queue_cap_with_concurrent_fan_out() -> None:
    """The 1000-message cap holds for every session even when many sessions are notified concurrently.

    Regression guard for the check-then-act race the DIFF-4 remediation must not introduce:
    each session has an independent queue key, so concurrent gather across *different* sessions
    must not let any single session's queue exceed _QUEUE_MAX.
    """
    store = InMemorySessionStore()
    mcp = MCPRouter(session_store=store, stateful=True)
    full_session = await store.create(protocol_version="2025-06-18", client_info={}, capabilities={})
    full_session.message_queue.extend({"jsonrpc": "2.0", "method": "filler"} for _ in range(1000))
    await store.update(full_session)
    other_sessions = [
        await store.create(protocol_version="2025-06-18", client_info={}, capabilities={}) for _ in range(10)
    ]

    await mcp.notify_tools_list_changed()

    full_messages = await store.dequeue_messages(full_session.session_id)
    assert len(full_messages) == 1000
    for session in other_sessions:
        messages = await store.dequeue_messages(session.session_id)
        assert messages == [{"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_notify_resource_updated_continues_after_one_subscriber_enqueue_fails() -> None:
    """A single subscriber's enqueue failure does not abort delivery to other subscribers."""
    store = _FlakyEnqueueSessionStore(failing_session_id="")
    failing_subscriber = await store.create(protocol_version="2025-06-18", client_info={}, capabilities={})
    store._failing_session_id = failing_subscriber.session_id
    healthy_subscriber = await store.create(protocol_version="2025-06-18", client_info={}, capabilities={})
    failing_subscriber.subscriptions.add("file:///doc.txt")
    healthy_subscriber.subscriptions.add("file:///doc.txt")
    await store.update(failing_subscriber)
    await store.update(healthy_subscriber)
    mcp = MCPRouter(session_store=store, stateful=True)

    await mcp.notify_resource_updated("file:///doc.txt")

    healthy_messages = await store.dequeue_messages(healthy_subscriber.session_id)
    assert healthy_messages == [
        {
            "jsonrpc": "2.0",
            "method": "notifications/resources/updated",
            "params": {"uri": "file:///doc.txt"},
        }
    ]


# ---------------------------------------------------------------------------
# Wire method-name literals stay snake_case; capability flags stay camelCase
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_all_four_notification_methods_use_snake_case_wire_names() -> None:
    """Each notify_* method enqueues its exact snake_case wire method name."""
    store = InMemorySessionStore()
    mcp = MCPRouter(session_store=store, stateful=True)
    session = await store.create(protocol_version="2025-06-18", client_info={}, capabilities={})
    session.subscriptions.add("file:///doc.txt")
    await store.update(session)

    await mcp.notify_tools_list_changed()
    await mcp.notify_resources_list_changed()
    await mcp.notify_prompts_list_changed()
    await mcp.notify_resource_updated("file:///doc.txt")

    messages = await store.dequeue_messages(session.session_id)
    methods = [message["method"] for message in messages]
    assert methods == [
        "notifications/tools/list_changed",
        "notifications/resources/list_changed",
        "notifications/prompts/list_changed",
        "notifications/resources/updated",
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_initialize_capability_flags_stay_camel_case() -> None:
    """The initialize response advertises listChanged in camelCase, never snake_case.

    Notification method names (asserted above) and capability sub-fields use
    different, non-convergent casing conventions by design.
    """
    store = InMemorySessionStore()
    mcp = MCPRouter(session_store=store, stateful=True)
    app = FastAPI()
    app.include_router(mcp, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18", "clientInfo": {}, "capabilities": {}},
            },
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 200
    capabilities = response.json()["result"]["capabilities"]
    assert capabilities["tools"] == {"listChanged": True}
    assert "list_changed" not in capabilities["tools"]


# ---------------------------------------------------------------------------
# Boundary: message queue cap is silently respected, no unbounded buffer
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_notify_drops_silently_at_thousand_message_queue_cap() -> None:
    """Enqueueing beyond the 1000-message cap drops the notification with no exception."""
    store = InMemorySessionStore()
    mcp = MCPRouter(session_store=store, stateful=True)
    session = await store.create(protocol_version="2025-06-18", client_info={}, capabilities={})
    session.message_queue.extend({"jsonrpc": "2.0", "method": "filler"} for _ in range(1000))
    await store.update(session)

    await mcp.notify_tools_list_changed()

    messages = await store.dequeue_messages(session.session_id)
    assert len(messages) == 1000
    assert all(message["method"] == "filler" for message in messages)


# ---------------------------------------------------------------------------
# Integration: notification reaches an active SSE stream within one dequeue cycle
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_notification_reaches_active_sse_stream_within_one_dequeue_cycle() -> None:
    """A notify_* call is visible on an active SSE stream without a new polling path.

    Bounds delivery with a generous asyncio.wait_for timeout (well above the
    documented 1s dequeue cycle) instead of sleeping a fixed 1s and racing
    the poll loop, avoiding the flake pattern already present elsewhere in
    this suite (see test_session_store.py::test_expired_session_returns_none).
    """

    async def auth_validator(api_key: str | None, bearer_token: str | None) -> bool:
        return bearer_token is not None

    store = InMemorySessionStore()
    mcp = MCPRouter(
        session_store=store,
        stateful=True,
        legacy_sse=True,
        auth_validator=auth_validator,
    )
    app = FastAPI()
    app.include_router(mcp, prefix="/mcp")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as init_client:
        init_response = await init_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18", "clientInfo": {}, "capabilities": {}},
            },
            headers={"Authorization": "Bearer test-token"},
        )
        assert init_response.status_code == 200, f"initialize failed: {init_response.text}"
        session_id = init_response.headers["Mcp-Session-Id"]

    capture = SseCapture(app)
    stream_transport = httpx.ASGITransport(app=capture)
    async with httpx.AsyncClient(transport=stream_transport, base_url="http://test") as stream_client:
        stream_task = asyncio.create_task(
            stream_client.get(
                "/mcp",
                headers={
                    "Authorization": "Bearer test-token",
                    "Mcp-Session-Id": session_id,
                },
            )
        )
        try:
            await asyncio.wait_for(capture.headers_received.wait(), timeout=5.0)
            assert capture.status_code == 200

            start = time.monotonic()
            await mcp.notify_tools_list_changed()

            async def _wait_for_delivery() -> None:
                while "notifications/tools/list_changed" not in "".join(capture.chunks):
                    await asyncio.sleep(0.05)

            await asyncio.wait_for(_wait_for_delivery(), timeout=5.0)
            elapsed = time.monotonic() - start
            assert elapsed < 2.0, f"Notification took {elapsed:.2f}s, expected ~1s dequeue cycle"
        finally:
            stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await stream_task

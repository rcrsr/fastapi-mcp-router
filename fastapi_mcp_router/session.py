"""Session storage abstractions for stateful MCP connections.

This module defines the SessionStore ABC and InMemorySessionStore implementation
for managing MCP session lifecycle in stateful (SSE streaming) mode.

Sessions track protocol negotiation state and a bounded message queue for
server-to-client notifications. The InMemorySessionStore suits single-instance
deployments; multi-instance deployments must provide a custom SessionStore.

Also provides SamplingManager for server-to-client LLM sampling requests,
RootsManager for tracking server operation boundaries, and MCPLoggingHandler
for sending log messages to connected clients.

Example::

    store = InMemorySessionStore(ttl_seconds=3600)
    session = await store.create(
        protocol_version="2025-06-18",
        client_info={"name": "claude"},
        capabilities={},
    )
    await store.enqueue_message(session.session_id, {"method": "ping"})
    messages = await store.dequeue_messages(session.session_id)
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

from fastapi_mcp_router.exceptions import MCPError

try:
    import redis.asyncio as _aioredis_runtime  # type: ignore[import-untyped]
except ImportError:
    _aioredis_runtime = None


class _AsyncRedisPipeline(Protocol):
    """Structural protocol for the redis.asyncio pipeline object used by RedisSessionStore."""

    def lrange(self, name: str, start: int, end: int) -> object: ...
    def delete(self, *names: str) -> object: ...
    async def execute(self) -> list[object]: ...


@runtime_checkable
class _AsyncRedisClient(Protocol):
    """Structural protocol for the redis.asyncio.Redis interface used by RedisSessionStore."""

    def set(self, name: str, value: str) -> Awaitable[object]: ...
    def get(self, name: str) -> Awaitable[str | bytes | None]: ...
    def expire(self, name: str, time: int) -> Awaitable[object]: ...
    def delete(self, *names: str) -> Awaitable[object]: ...
    def llen(self, name: str) -> Awaitable[int]: ...
    def lpush(self, name: str, *values: str) -> Awaitable[object]: ...
    def lrange(self, name: str, start: int, end: int) -> Awaitable[list[str | bytes]]: ...
    def pipeline(self) -> _AsyncRedisPipeline: ...


logger = logging.getLogger(__name__)

_QUEUE_MAX = 1000
_SUBSCRIPTIONS_MAX = 100


@dataclass
class Session:
    """MCP session state for a single connected client.

    Tracks protocol negotiation, client capabilities, and queued
    server-to-client messages for a single stateful connection.

    Attributes:
        session_id: UUID4 string, immutable after creation.
        created_at: UTC datetime when the session was created, immutable.
        last_activity: UTC datetime of the last access; updated on each get().
        protocol_version: Negotiated MCP protocol version string.
        client_info: Client capabilities dict from the initialize request.
        capabilities: Negotiated server capabilities dict.
        message_queue: Queued server-to-client messages, max 1000.
        subscriptions: Resource URIs with active subscriptions, max 100.
    """

    session_id: str
    created_at: datetime
    last_activity: datetime
    protocol_version: str
    client_info: dict
    capabilities: dict
    message_queue: list[dict] = field(default_factory=list)
    subscriptions: set[str] = field(default_factory=set)


class SessionStore(ABC):
    """Abstract base class for MCP session persistence.

    Implementations must be safe for concurrent async access. All methods
    are async to allow I/O-bound backends (Redis, database) without interface
    changes.

    Example::

        class RedisSessionStore(SessionStore):
            async def create(self, protocol_version, client_info, capabilities):
                ...
    """

    @abstractmethod
    async def create(
        self,
        protocol_version: str,
        client_info: dict,
        capabilities: dict,
    ) -> Session:
        """Create a new session with a generated UUID4 session_id.

        Args:
            protocol_version: Negotiated MCP protocol version string.
            client_info: Client capabilities dict from the initialize request.
            capabilities: Negotiated server capabilities dict.

        Returns:
            Newly created Session with generated session_id and UTC timestamps.
        """
        ...

    @abstractmethod
    async def get(self, session_id: str) -> Session | None:
        """Retrieve a session by ID, returning None if expired or absent.

        Updates last_activity on the session when found and not expired.

        Args:
            session_id: UUID4 string identifying the session.

        Returns:
            Session if found and not expired, None otherwise.
        """
        ...

    @abstractmethod
    async def update(self, session: Session) -> None:
        """Persist changes to an existing session.

        Args:
            session: Session object with updated fields to persist.
        """
        ...

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Remove a session from the store.

        Args:
            session_id: UUID4 string identifying the session to delete.
        """
        ...

    @abstractmethod
    async def enqueue_message(self, session_id: str, message: dict) -> None:
        """Append a message to the session's queue.

        Silently drops the message if the queue already holds 1000 messages.

        Args:
            session_id: UUID4 string identifying the target session.
            message: JSON-RPC notification dict to enqueue.
        """
        ...

    @abstractmethod
    async def dequeue_messages(self, session_id: str) -> list[dict]:
        """Return all queued messages and clear the queue atomically.

        Args:
            session_id: UUID4 string identifying the target session.

        Returns:
            List of queued message dicts; empty list if none queued or session absent.
        """
        ...


class InMemorySessionStore(SessionStore):
    """In-memory SessionStore with TTL-based expiration.

    Stores sessions in a dict protected by an asyncio.Lock. Suitable for
    single-instance deployments only; all state is lost on restart.

    Attributes:
        ttl_seconds: Seconds of inactivity before a session expires.

    Example::

        store = InMemorySessionStore(ttl_seconds=1800)
        session = await store.create("2025-06-18", {}, {})
        found = await store.get(session.session_id)
        assert found is not None
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        """Initialize the store with a configurable TTL.

        Args:
            ttl_seconds: Inactivity TTL in seconds (default 3600).
        """
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        protocol_version: str,
        client_info: dict,
        capabilities: dict,
    ) -> Session:
        """Create and persist a new session.

        Args:
            protocol_version: Negotiated MCP protocol version string.
            client_info: Client capabilities dict from the initialize request.
            capabilities: Negotiated server capabilities dict.

        Returns:
            Newly created Session stored in memory.
        """
        now = datetime.now(UTC)
        session = Session(
            session_id=str(uuid.uuid4()),
            created_at=now,
            last_activity=now,
            protocol_version=protocol_version,
            client_info=client_info,
            capabilities=capabilities,
        )
        async with self._lock:
            self._sessions[session.session_id] = session
        return session

    async def get(self, session_id: str) -> Session | None:
        """Retrieve a session, returning None if expired or absent.

        Updates last_activity when the session is found and still valid.

        Args:
            session_id: UUID4 string identifying the session.

        Returns:
            Session if present and not expired, None otherwise.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            now = datetime.now(UTC)
            if now - session.last_activity > timedelta(seconds=self.ttl_seconds):
                del self._sessions[session_id]
                logger.debug("Session expired and removed: %s", session_id)
                return None
            session.last_activity = now
            return session

    async def update(self, session: Session) -> None:
        """Persist changes to an existing session.

        Args:
            session: Session object with updated fields.
        """
        async with self._lock:
            self._sessions[session.session_id] = session

    async def delete(self, session_id: str) -> None:
        """Remove a session from the store.

        Args:
            session_id: UUID4 string identifying the session to delete.
        """
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def enqueue_message(self, session_id: str, message: dict) -> None:
        """Append a message to the session's queue, dropping if full.

        Silently drops the message when the queue already holds 1000 messages.

        Args:
            session_id: UUID4 string identifying the target session.
            message: JSON-RPC notification dict to enqueue.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            if len(session.message_queue) >= _QUEUE_MAX:
                logger.warning("Message queue full for session %s; dropping message", session_id)
                return
            session.message_queue.append(message)

    async def dequeue_messages(self, session_id: str) -> list[dict]:
        """Return all queued messages and clear the queue atomically.

        Args:
            session_id: UUID4 string identifying the target session.

        Returns:
            List of queued message dicts; empty list if none queued or session absent.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []
            messages = session.message_queue
            session.message_queue = []
            return messages


class RedisSessionStore(SessionStore):
    """Redis-backed SessionStore with TTL-based expiration.

    Stores session data as a JSON string at key ``mcp:session:{session_id}``
    and the message queue as a Redis list at ``mcp:queue:{session_id}``.
    TTL is set on both keys after every create or update operation.

    redis.asyncio is an optional dependency. Import this class only when
    redis-py is installed; constructing it without redis installed raises
    ``RuntimeError``.

    Attributes:
        ttl_seconds: Seconds until a session key expires in Redis.

    Example::

        import redis.asyncio as aioredis
        client = aioredis.Redis.from_url("redis://localhost")
        store = RedisSessionStore(redis_client=client, ttl_seconds=7200)
        session = await store.create("2025-06-18", {}, {})
        found = await store.get(session.session_id)
        assert found is not None
    """

    def __init__(self, redis_client: object, ttl_seconds: int = 7200) -> None:
        """Initialize with an existing async Redis connection.

        Args:
            redis_client: A ``redis.asyncio.Redis`` instance (already connected).
            ttl_seconds: Session key TTL in seconds (default 7200).

        Raises:
            RuntimeError: If redis.asyncio is not installed.
        """
        if _aioredis_runtime is None:
            raise RuntimeError("redis-py is required for RedisSessionStore. Install it with: pip install redis")
        self._redis: _AsyncRedisClient = redis_client  # type: ignore[assignment]
        self.ttl_seconds = ttl_seconds

    def _session_key(self, session_id: str) -> str:
        return f"mcp:session:{session_id}"

    def _queue_key(self, session_id: str) -> str:
        return f"mcp:queue:{session_id}"

    def _serialize(self, session: Session) -> str:
        return json.dumps(
            {
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "protocol_version": session.protocol_version,
                "client_info": session.client_info,
                "capabilities": session.capabilities,
                "subscriptions": list(session.subscriptions),
            }
        )

    def _deserialize(self, raw: str) -> Session:
        data = json.loads(raw)
        return Session(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            protocol_version=data["protocol_version"],
            client_info=data["client_info"],
            capabilities=data["capabilities"],
            subscriptions=set(data.get("subscriptions", [])),
        )

    async def create(
        self,
        protocol_version: str,
        client_info: dict,
        capabilities: dict,
    ) -> Session:
        """Create and persist a new session in Redis.

        Args:
            protocol_version: Negotiated MCP protocol version string.
            client_info: Client capabilities dict from the initialize request.
            capabilities: Negotiated server capabilities dict.

        Returns:
            Newly created Session stored in Redis.

        Raises:
            MCPError: -32603 if the Redis operation fails.
        """
        now = datetime.now(UTC)
        session = Session(
            session_id=str(uuid.uuid4()),
            created_at=now,
            last_activity=now,
            protocol_version=protocol_version,
            client_info=client_info,
            capabilities=capabilities,
        )
        try:
            session_key = self._session_key(session.session_id)
            await self._redis.set(session_key, self._serialize(session))
            await self._redis.expire(session_key, self.ttl_seconds)
        except Exception as e:
            raise MCPError(code=-32603, message=f"Redis error creating session: {e}") from e
        return session

    async def get(self, session_id: str) -> Session | None:
        """Retrieve a session from Redis, returning None if absent or expired.

        Updates last_activity when the session is found.

        Args:
            session_id: UUID4 string identifying the session.

        Returns:
            Session if present, None otherwise.

        Raises:
            MCPError: -32603 if the Redis operation fails.
        """
        try:
            raw = await self._redis.get(self._session_key(session_id))
        except Exception as e:
            raise MCPError(code=-32603, message=f"Redis error fetching session: {e}") from e
        if raw is None:
            return None
        session = self._deserialize(raw if isinstance(raw, str) else raw.decode())
        session.last_activity = datetime.now(UTC)
        try:
            session_key = self._session_key(session_id)
            await self._redis.set(session_key, self._serialize(session))
            await self._redis.expire(session_key, self.ttl_seconds)
        except Exception as e:
            raise MCPError(code=-32603, message=f"Redis error updating session activity: {e}") from e
        return session

    async def update(self, session: Session) -> None:
        """Persist changes to an existing session in Redis.

        Args:
            session: Session object with updated fields to persist.

        Raises:
            MCPError: -32603 if the Redis operation fails.
        """
        try:
            session_key = self._session_key(session.session_id)
            await self._redis.set(session_key, self._serialize(session))
            await self._redis.expire(session_key, self.ttl_seconds)
        except Exception as e:
            raise MCPError(code=-32603, message=f"Redis error updating session: {e}") from e

    async def delete(self, session_id: str) -> None:
        """Remove a session and its message queue from Redis.

        Args:
            session_id: UUID4 string identifying the session to delete.

        Raises:
            MCPError: -32603 if the Redis operation fails.
        """
        try:
            await self._redis.delete(
                self._session_key(session_id),
                self._queue_key(session_id),
            )
        except Exception as e:
            raise MCPError(code=-32603, message=f"Redis error deleting session: {e}") from e

    async def enqueue_message(self, session_id: str, message: dict) -> None:
        """Append a message to the session's Redis list queue, dropping if full.

        Silently drops the message when the queue already holds 1000 messages.

        Args:
            session_id: UUID4 string identifying the target session.
            message: JSON-RPC notification dict to enqueue.

        Raises:
            MCPError: -32603 if the Redis operation fails.
        """
        queue_key = self._queue_key(session_id)
        try:
            current_len = await self._redis.llen(queue_key)
            if current_len >= _QUEUE_MAX:
                logger.warning("Message queue full for session %s; dropping message", session_id)
                return
            await self._redis.lpush(queue_key, json.dumps(message))
            await self._redis.expire(queue_key, self.ttl_seconds)
        except Exception as e:
            raise MCPError(code=-32603, message=f"Redis error enqueuing message: {e}") from e

    async def dequeue_messages(self, session_id: str) -> list[dict]:
        """Return all queued messages and clear the queue atomically.

        Uses a Redis pipeline to issue LRANGE and DEL in a single round-trip,
        preventing a concurrent reader from observing the same messages between
        the two commands.

        Args:
            session_id: UUID4 string identifying the target session.

        Returns:
            List of queued message dicts; empty list if none queued or session absent.

        Raises:
            MCPError: -32603 if the Redis operation fails.
        """
        queue_key = self._queue_key(session_id)
        try:
            pipe = self._redis.pipeline()
            pipe.lrange(queue_key, 0, -1)
            pipe.delete(queue_key)
            results = await pipe.execute()
            raw_messages: list[str | bytes] = results[0]  # type: ignore[assignment]
            if not raw_messages:
                return []
            raw_messages.reverse()
            return [json.loads(m if isinstance(m, str) else m.decode()) for m in raw_messages]
        except MCPError:
            raise
        except Exception as e:
            raise MCPError(code=-32603, message=f"Redis error dequeuing messages: {e}") from e


class ProgressTracker:
    """Progress and cancellation management for in-flight tool requests.

    Enqueues notifications/progress JSON-RPC notifications via a SessionStore
    and tracks cancelled request IDs so tool handlers can check and exit early.

    Attributes:
        _session_store: SessionStore used to enqueue progress notifications.
        _cancelled: Set of request IDs that have been requested for cancellation.

    Example::

        tracker = ProgressTracker(session_store=store)
        await tracker.report_progress("sess-1", "req-1", 5, 10, "halfway")
        tracker.request_cancellation("req-1")
        assert tracker.is_cancelled("req-1")
        tracker.clear_cancellation("req-1")
    """

    def __init__(self, session_store: SessionStore) -> None:
        """Initialize with the SessionStore used for enqueuing notifications.

        Args:
            session_store: SessionStore instance for enqueueing progress messages.
        """
        self._session_store = session_store
        self._cancelled: set[str] = set()

    async def report_progress(
        self,
        session_id: str,
        request_id: str,
        current: int,
        total: int,
        message: str | None,
    ) -> None:
        """Enqueue a notifications/progress JSON-RPC notification.

        Builds the notification dict and enqueues it via the SessionStore.
        If the enqueue raises an exception, the error is logged and not
        propagated, so the calling tool continues execution.

        Args:
            session_id: UUID4 string identifying the target session.
            request_id: JSON-RPC request ID used as the progressToken.
            current: Number of units of work completed.
            total: Total units of work expected.
            message: Optional human-readable status message; omitted when None.

        Returns:
            None
        """
        params: dict[str, object] = {
            "progressToken": request_id,
            "progress": current,
            "total": total,
        }
        if message is not None:
            params["message"] = message
        notification: dict[str, object] = {
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": params,
        }
        try:
            await self._session_store.enqueue_message(session_id, notification)
        except Exception:
            logger.exception(
                "Failed to enqueue progress notification for session %s request %s",
                session_id,
                request_id,
            )

    def request_cancellation(self, request_id: str) -> None:
        """Mark a request as cancelled.

        Args:
            request_id: JSON-RPC request ID to cancel.

        Returns:
            None
        """
        self._cancelled.add(request_id)

    def is_cancelled(self, request_id: str) -> bool:
        """Return True if the request has been marked for cancellation.

        Args:
            request_id: JSON-RPC request ID to check.

        Returns:
            True if request_id is in the cancelled set, False otherwise.
        """
        return request_id in self._cancelled

    def clear_cancellation(self, request_id: str) -> None:
        """Remove a request ID from the cancelled set.

        Args:
            request_id: JSON-RPC request ID to remove.

        Returns:
            None
        """
        self._cancelled.discard(request_id)


_SAMPLING_TIMEOUT = 60.0

_LOG_LEVEL_PRIORITY: dict[str, int] = {
    "debug": 0,
    "info": 1,
    "notice": 2,
    "warning": 3,
    "error": 4,
    "critical": 5,
    "alert": 6,
    "emergency": 7,
}


class SamplingManager:
    """Server-to-client LLM sampling request manager.

    Enqueues sampling/createMessage requests via a SessionStore and correlates
    client responses to waiting callers using asyncio.Future objects keyed by
    request ID.

    Attributes:
        _session_store: SessionStore used to enqueue sampling requests.
        _sampling_enabled: Whether sampling is permitted on this server.
        _pending: Map of request_id to asyncio.Future awaiting client response.

    Example::

        manager = SamplingManager(session_store=store, sampling_enabled=True)
        response = await manager.create_message(
            session_id="sess-1",
            messages=[{"role": "user", "content": {"type": "text", "text": "Hello"}}],
        )
        assert response["model"]
    """

    def __init__(
        self,
        session_store: SessionStore | None,
        sampling_enabled: bool = False,
    ) -> None:
        """Initialize with a SessionStore and sampling flag.

        Args:
            session_store: SessionStore instance for enqueueing requests. Must
                not be None when sampling is used (EC-19).
            sampling_enabled: Whether sampling is permitted. Must be True when
                create_message is called (EC-20).

        Raises:
            MCPError: -32601 if session_store is None (stateful mode required).
        """
        if session_store is None:
            raise MCPError(
                code=-32601,
                message="Sampling requires stateful mode (session_store must not be None)",
            )
        self._session_store = session_store
        self._sampling_enabled = sampling_enabled
        self._pending: dict[str, asyncio.Future] = {}

    async def create_message(
        self,
        session_id: str,
        messages: list[dict],
        model_preferences: dict | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> dict:
        """Send a sampling/createMessage request to the client and await the response.

        Generates a UUID4 request_id, enqueues the request via the SessionStore,
        registers a Future, and awaits it with a 60-second timeout.

        Args:
            session_id: UUID4 string identifying the target session.
            messages: List of message dicts for the sampling request.
            model_preferences: Optional model preference hints dict.
            system_prompt: Optional system prompt string.
            temperature: Optional sampling temperature float.
            max_tokens: Optional maximum tokens integer.
            stop_sequences: Optional list of stop sequence strings.

        Returns:
            SamplingResponse dict with model, role, content, and stop_reason fields.

        Raises:
            MCPError: -32601 if sampling_enabled is False (EC-20).
            MCPError: -32603 if the client does not respond within 60 seconds (EC-21).
        """
        if not self._sampling_enabled:
            raise MCPError(
                code=-32601,
                message="Sampling is not enabled on this server",
            )
        request_id = str(uuid.uuid4())
        message: dict[str, object] = {
            "jsonrpc": "2.0",
            "method": "sampling/createMessage",
            "id": request_id,
            "params": {
                "messages": messages,
                "modelPreferences": model_preferences,
                "systemPrompt": system_prompt,
                "temperature": temperature,
                "maxTokens": max_tokens,
                "stopSequences": stop_sequences,
            },
        }
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._session_store.enqueue_message(session_id, message)
            return await asyncio.wait_for(future, timeout=_SAMPLING_TIMEOUT)
        except TimeoutError as err:
            raise MCPError(
                code=-32603,
                message=f"Sampling request timed out after {_SAMPLING_TIMEOUT:.0f}s",
            ) from err
        finally:
            self._pending.pop(request_id, None)

    def handle_response(self, request_id: str, response: dict) -> None:
        """Correlate a client sampling response to the waiting create_message call.

        Looks up the Future registered for request_id and sets its result,
        resolving the awaiting create_message call. Silently ignores unknown IDs.

        Args:
            request_id: Request ID matching the original sampling/createMessage.
            response: Response dict from the client containing model, role, content.

        Returns:
            None
        """
        future = self._pending.get(request_id)
        if future is not None and not future.done():
            future.set_result(response)


class RootsManager:
    """Server operation boundary definitions.

    Maintains a list of Root entries that define the URIs the server operates
    on. Roots are registered via add_root and retrieved via list_roots.

    Attributes:
        _roots: Internal list of Root dicts representing registered boundaries.

    Example::

        manager = RootsManager()
        manager.add_root(uri="file:///workspace", name="Workspace")
        roots = manager.list_roots()
        assert roots[0]["uri"] == "file:///workspace"
    """

    def __init__(self) -> None:
        """Initialize with an empty roots list."""
        self._roots: list[dict] = []

    def add_root(self, uri: str, name: str | None = None) -> None:
        """Register a new operation boundary by URI.

        Args:
            uri: URI string identifying the root boundary.
            name: Optional human-readable name for the root.

        Returns:
            None
        """
        root: dict[str, object] = {"uri": uri}
        if name is not None:
            root["name"] = name
        self._roots.append(root)

    def list_roots(self) -> list[dict]:
        """Return a copy of all registered roots.

        Returns:
            List of root dicts, each containing uri and optional name.
        """
        return list(self._roots)


class MCPLoggingHandler:
    """MCP logging protocol handler for sending log messages to connected clients.

    Enqueues notifications/message JSON-RPC notifications via a SessionStore.
    Maintains per-session minimum log levels and filters messages below that level.

    Log levels in ascending priority: debug, info, notice, warning, error,
    critical, alert, emergency. Default minimum level is info.

    Attributes:
        _session_store: SessionStore used to enqueue log notifications.
        _levels: Per-session minimum log level strings.

    Example::

        handler = MCPLoggingHandler(session_store=store)
        handler.set_level("sess-1", "warning")
        await handler.log_message("sess-1", "debug", "myapp", "verbose detail")  # no-op
        await handler.log_message("sess-1", "error", "myapp", "something failed")  # sent
    """

    _DEFAULT_LEVEL = "info"

    def __init__(self, session_store: SessionStore | None) -> None:
        """Initialize with a SessionStore.

        Args:
            session_store: SessionStore instance for enqueueing log notifications.
                Must not be None (EC-23).

        Raises:
            MCPError: -32601 if session_store is None (stateful mode required).
        """
        if session_store is None:
            raise MCPError(
                code=-32601,
                message="Logging requires stateful mode (session_store must not be None)",
            )
        self._session_store = session_store
        self._levels: dict[str, str] = {}

    def _validate_log_level(self, level: str) -> str:
        """Normalise and validate a log level string.

        Args:
            level: Log level string to validate (case-insensitive).

        Returns:
            Normalised (lowercase) log level string.

        Raises:
            MCPError: -32602 if level is not a recognised log level string (EC-22).
        """
        normalised = level.lower() if isinstance(level, str) else str(level).lower()
        if normalised not in _LOG_LEVEL_PRIORITY:
            valid = ", ".join(_LOG_LEVEL_PRIORITY)
            raise MCPError(
                code=-32602,
                message=f"Invalid log level: {level!r}. Valid levels: {valid}",
            )
        return normalised

    def set_level(self, session_id: str, level: str) -> None:
        """Set the minimum log level for a session.

        Args:
            session_id: UUID4 string identifying the target session.
            level: Log level string (debug, info, notice, warning, error,
                critical, alert, emergency).

        Returns:
            None

        Raises:
            MCPError: -32602 if level is not a recognised log level string (EC-22).
        """
        normalised = self._validate_log_level(level)
        self._levels[session_id] = normalised

    async def log_message(
        self,
        session_id: str,
        level: str,
        logger: str,
        data: object,
    ) -> None:
        """Send a log notification to the client if level meets the session minimum.

        Builds a notifications/message notification and enqueues it via the
        SessionStore. Does nothing if the message level is below the session's
        configured minimum level.

        Args:
            session_id: UUID4 string identifying the target session.
            level: Log level string for this message.
            logger: Logger name string identifying the source component.
            data: Arbitrary log data payload (serialisable object).

        Returns:
            None

        Raises:
            MCPError: -32602 if level is not a recognised log level string (EC-22).
        """
        normalised = self._validate_log_level(level)
        min_level = self._levels.get(session_id, self._DEFAULT_LEVEL)
        if _LOG_LEVEL_PRIORITY[normalised] < _LOG_LEVEL_PRIORITY[min_level]:
            return
        notification: dict[str, object] = {
            "jsonrpc": "2.0",
            "method": "notifications/message",
            "params": {
                "level": normalised,
                "logger": logger,
                "data": data,
            },
        }
        try:
            await self._session_store.enqueue_message(session_id, notification)
        except Exception:
            logging.getLogger(__name__).exception(
                "Failed to enqueue log notification for session %s",
                session_id,
            )

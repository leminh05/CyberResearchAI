"""In-memory request limiter and conversation store (one server process)."""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable
from uuid import UUID, uuid4

from cyberresearch.rag.engine import RAGEngine


class RateLimiter:
    """Sliding window: at most ``limit`` requests per ``window`` seconds per client."""

    def __init__(self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.limit, self.window, self.clock = limit, window, clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def retry_after(self, client: str) -> int:
        """Record a request; 0 if allowed, otherwise the seconds to wait."""
        now = self.clock()
        with self._lock:
            # Forget idle clients so the table cannot grow without bound.
            for key in [k for k, hits in self._hits.items() if hits[-1] <= now - self.window]:
                del self._hits[key]
            hits = self._hits.setdefault(client, deque())
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(1, math.ceil(hits[0] + self.window - now))
            hits.append(now)
            return 0


@dataclass
class Conversation:
    engine: RAGEngine  # holds the follow-up history of this conversation
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_used: float = 0.0


class ConversationStore:
    """Conversations by unguessable id, dropped after ``ttl`` seconds or when full."""

    def __init__(self, new_engine: Callable[[], RAGEngine], max_items: int, ttl: float,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.new_engine, self.max_items, self.ttl, self.clock = new_engine, max_items, ttl, clock
        self._items: OrderedDict[UUID, Conversation] = OrderedDict()
        self._lock = threading.Lock()

    def _expire(self, now: float) -> None:
        while self._items and next(iter(self._items.values())).last_used <= now - self.ttl:
            self._items.popitem(last=False)

    def create(self) -> tuple[UUID, Conversation]:
        now = self.clock()
        conversation_id, conversation = uuid4(), Conversation(self.new_engine(), last_used=now)
        with self._lock:
            self._expire(now)
            self._items[conversation_id] = conversation
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)
        return conversation_id, conversation

    def get(self, conversation_id: UUID) -> Conversation | None:
        now = self.clock()
        with self._lock:
            self._expire(now)
            conversation = self._items.get(conversation_id)
            if conversation:
                conversation.last_used = now
                self._items.move_to_end(conversation_id)
            return conversation

    def delete(self, conversation_id: UUID) -> bool:
        with self._lock:
            return self._items.pop(conversation_id, None) is not None

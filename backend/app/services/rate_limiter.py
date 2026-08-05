import threading
import time
from collections import deque

class GlobalRateLimiter:
    def __init__(self, max_calls, period_seconds = 60.0):
        self.max_calls = max_calls
        self.period = period_seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def allow(self):
        now = time.monotonic()
        with self._lock:
            while self._timestamps and now - self._timestamps[0] > self.period:
                self._timestamps.popleft()
            if len(self._timestamps) >= self.max_calls:
                return False
            self._timestamps.append(now)
            return True
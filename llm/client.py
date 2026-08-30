"""Model client: OpenAI-compatible chat endpoint, or a mock for offline smoke tests.

Fill config.yaml (see config.example.yaml) with your model name, api_key and
base_url; everything else in llm/ reads from there. Uses only the standard
library, so no SDK install is required.

Reliability features (paid-run armor):
  - exponential backoff with jitter on HTTP 429 / 5xx and network timeouts
    (client and 4xx-other-than-429 errors are raised immediately: they are
    config bugs, not transient failures);
  - max_calls cost fuse: raises CallLimitExceeded so the run stops cleanly,
    letting the runner checkpoint everything completed so far;
  - optional per-call JSONL log (timestamp, latency, tokens, status) for cost
    accounting and post-mortems.
"""
import json
import random
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path


class CallLimitExceeded(RuntimeError):
    """Raised when the client's max_calls fuse trips. Runners must catch this,
    checkpoint completed episodes and exit cleanly."""


class ChatClient:
    def __init__(self, model, api_key, base_url, temperature=0.0, max_tokens=64,
                 timeout=120, max_retries=5, max_calls=None, log_path=None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_calls = max_calls
        self.n_calls = 0
        self.n_tokens = 0
        self.n_errors = 0
        self._log = open(log_path, 'a') if log_path else None

    def close(self):
        if self._log:
            self._log.close()
            self._log = None

    def _write_log(self, rec):
        if self._log:
            self._log.write(json.dumps(rec) + '\n')
            self._log.flush()

    def chat(self, system, user, max_tokens=None):
        if self.max_calls is not None and self.n_calls >= self.max_calls:
            raise CallLimitExceeded(
                f'max_calls fuse tripped at {self.max_calls} calls')
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }).encode()
        t0 = time.time()
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    self.base_url + "/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {self.api_key}"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                self.n_calls += 1
                usage = data.get("usage") or {}
                self.n_tokens += usage.get("total_tokens", 0)
                self._write_log(dict(ts=time.time(), ok=True,
                                     latency_s=round(time.time() - t0, 3),
                                     total_tokens=usage.get("total_tokens", 0),
                                     attempt=attempt))
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                last_err = e
                retryable = e.code == 429 or 500 <= e.code < 600
                if not retryable:
                    self.n_errors += 1
                    self._write_log(dict(ts=time.time(), ok=False,
                                         error=f'HTTP {e.code}', fatal=True))
                    raise  # config/auth bug: fail fast, do not burn retries
            except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as e:
                last_err = e
            # transient failure: backoff with jitter, then retry
            wait = min(2 ** attempt, 30) + random.random()
            self._write_log(dict(ts=time.time(), ok=False, attempt=attempt,
                                 error=type(last_err).__name__ + ': ' + str(last_err)[:200],
                                 backoff_s=round(wait, 2)))
            time.sleep(wait)
        self.n_errors += 1
        raise RuntimeError(f'chat() failed after {self.max_retries + 1} attempts: '
                           f'{last_err}')


class MockClient:
    """Offline smoke-test client: picks a random legal-looking move.
    Never touches the network; lets you validate the whole engine end to end."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.n_calls = 0
        self.n_tokens = 0
        self.n_errors = 0
        self.max_calls = None

    def close(self):
        pass

    def chat(self, system, user, max_tokens=None):
        self.n_calls += 1
        if system.startswith('You just walked'):
            # authoring call: echo one valid note from the first junction listed
            for line in user.splitlines():
                if line.startswith('JCT '):
                    cell = line.split()[1]
                    d = line.split('OPEN=', 1)[1][:1]
                    return f"NOTE: {cell} | {d} | mock note"
            return "NONE"
        legal = 'NSEW'
        for line in user.splitlines():
            if line.startswith('OPEN:'):
                legal = line.split(':', 1)[1].strip() or 'NSEW'
        return f"MOVE: {self.rng.choice(list(legal))}"


def from_config(cfg):
    m = cfg['model']
    if m.get('name', '').lower() == 'mock':
        return MockClient(seed=cfg.get('maze', {}).get('seed', 0))
    return ChatClient(model=m['name'], api_key=m['api_key'], base_url=m['base_url'],
                      temperature=m.get('temperature', 0.0),
                      max_tokens=m.get('max_tokens', 64),
                      timeout=m.get('timeout', 120),
                      max_retries=m.get('max_retries', 5),
                      max_calls=m.get('max_calls'),
                      log_path=m.get('log_path'))

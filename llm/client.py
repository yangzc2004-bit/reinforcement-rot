"""Model client: OpenAI-compatible chat endpoint, or a mock for offline smoke tests.

Fill config.yaml (see config.example.yaml) with your model name, api_key and
base_url; everything else in llm/ reads from there. Uses only the standard
library, so no SDK install is required.
"""
import json
import random
import urllib.request


class ChatClient:
    def __init__(self, model, api_key, base_url, temperature=0.0, max_tokens=64, timeout=120):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.n_calls = 0
        self.n_tokens = 0

    def chat(self, system, user):
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }).encode()
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
        return data["choices"][0]["message"]["content"]


class MockClient:
    """Offline smoke-test client: picks a random legal-looking move.
    Never touches the network; lets you validate the whole engine end to end."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.n_calls = 0
        self.n_tokens = 0

    def chat(self, system, user):
        self.n_calls += 1
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
                      max_tokens=m.get('max_tokens', 64))

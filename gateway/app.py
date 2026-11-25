"""CIUK26 FastAPI gateway: routes OpenAI-style streaming requests to llama-server pods.

What it does for every POST /v1/chat/completions:
  1. pick a backend pod (least-in-flight for short prompts, prefix-affinity for long ones)
  2. forward the request and stream the SSE bytes straight back (no buffering)
  3. if a pod dies before any token was sent, transparently retry on another pod
  4. record Prometheus metrics (TTFT, ITL, tokens, in-flight, errors) for GET /metrics

Config (environment variables):
  BACKENDS        static list for local dev, e.g. "localhost:8080,localhost:8081"
  BACKEND_DNS     headless Service name to discover pods in K8s (default: llama-worker-headless)
  BACKEND_PORT    llama-server port (default: 8080)
  ROUTING         "smart" (default) or "roundrobin"  -> lets you benchmark the difference
  CACHE_PROMPT    "true" / "false" / unset -> force llama-server's cache_prompt per request
                  (set "false" for phase2_baseline.json, "true" for the cached run)
  LONG_PROMPT_CHARS  prompts longer than this count as "long" (default: 3000)
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import os
import socket
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# ---------------------------------------------------------------- config
STATIC_BACKENDS = [b.strip() for b in os.getenv("BACKENDS", "").split(",") if b.strip()]
BACKEND_DNS = os.getenv("BACKEND_DNS", "llama-worker-headless")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8080"))
ROUTING = os.getenv("ROUTING", "smart")
CACHE_PROMPT = os.getenv("CACHE_PROMPT", "").lower()
LONG_PROMPT_CHARS = int(os.getenv("LONG_PROMPT_CHARS", "3000"))
PREFIX_CHARS = 2000          # how much of the prompt we hash for prefix affinity
AFFINITY_SLACK = 2           # abandon the "cached" pod if it's this much busier than the idlest
MAX_ATTEMPTS = 3
COOLDOWN_S = 5.0             # avoid a pod for this long after it fails

# ---------------------------------------------------------------- metrics
BUCKETS = (0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13, 20, 30)
REQUESTS = Counter("gateway_requests_total", "Requests handled", ["pool", "status"])
ERRORS = Counter("gateway_errors_total", "Failed requests", ["reason"])
RETRIES = Counter("gateway_retries_total", "Requests retried on another backend")
TTFT = Histogram("gateway_ttft_seconds", "Time to first token", ["pool"], buckets=BUCKETS)
ITL = Histogram("gateway_itl_seconds", "Inter-token latency",
                buckets=(0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1))
E2E = Histogram("gateway_request_duration_seconds", "End-to-end latency", ["pool"], buckets=BUCKETS)
TOKENS = Counter("gateway_output_tokens_total", "Completion tokens streamed", ["pool"])
INFLIGHT = Gauge("gateway_inflight", "In-flight requests per backend", ["backend"])
QUEUE = Gauge("gateway_inflight_total", "Total in-flight requests (queue depth)")
BACKENDS_UP = Gauge("gateway_backends_available", "Backends currently routable")

# ---------------------------------------------------------------- state
app = FastAPI()
client: httpx.AsyncClient
backends: list[str] = list(STATIC_BACKENDS)
inflight: dict[str, int] = {}
inflight_long: dict[str, int] = {}
bad_until: dict[str, float] = {}
rr = itertools.count()


def _discover() -> list[str]:
    """Headless Service DNS returns one A record per *ready* pod."""
    infos = socket.getaddrinfo(BACKEND_DNS, BACKEND_PORT, proto=socket.IPPROTO_TCP)
    return sorted({f"{i[4][0]}:{BACKEND_PORT}" for i in infos})


async def discovery_loop() -> None:
    global backends
    while True:
        try:
            found = await asyncio.to_thread(_discover)
            if found:
                backends = found
        except OSError:
            pass  # DNS hiccup (e.g. during a pod restart): keep the last known list
        BACKENDS_UP.set(len(routable()))
        await asyncio.sleep(2)


@app.on_event("startup")
async def startup() -> None:
    global client
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=2.0, read=120.0, write=10.0, pool=5.0),
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
    )
    if not STATIC_BACKENDS:
        asyncio.create_task(discovery_loop())
    else:
        BACKENDS_UP.set(len(backends))


@app.on_event("shutdown")
async def shutdown() -> None:
    await client.aclose()


# ---------------------------------------------------------------- routing
def routable(exclude: set[str] = frozenset()) -> list[str]:
    now = time.monotonic()
    good = [b for b in backends if b not in exclude and bad_until.get(b, 0) < now]
    return good or [b for b in backends if b not in exclude]  # all "bad"? try anyway


def prompt_text(body: dict) -> str:
    return "\n".join(str(m.get("content", "")) for m in body.get("messages", []))


def pick_backend(text: str, is_long: bool, exclude: set[str]) -> str | None:
    cands = routable(exclude)
    if not cands:
        return None
    if ROUTING == "roundrobin":
        return cands[next(rr) % len(cands)]

    load = lambda b: inflight.get(b, 0) + 2 * inflight_long.get(b, 0)  # long jobs weigh more

    if is_long:
        # Prefix affinity via rendezvous hashing: the same shared prefix always maps to
        # the same pod (so its KV cache is warm), and the mapping is stable when pods change.
        key = text[:PREFIX_CHARS].encode()
        ranked = sorted(cands, key=lambda b: hashlib.md5(key + b.encode()).digest(), reverse=True)
        preferred = ranked[0]
        idlest = min(cands, key=load)
        if load(preferred) - load(idlest) > AFFINITY_SLACK:
            return idlest  # cached pod is swamped: a cold prefill elsewhere is cheaper
        return preferred

    # Short prompts: least-in-flight, penalising pods busy with long prefills.
    return min(cands, key=lambda b: (load(b), next(rr) % 997))


# ---------------------------------------------------------------- endpoints
@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "backends": backends, "inflight": inflight}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    if CACHE_PROMPT in ("true", "false"):
        body["cache_prompt"] = CACHE_PROMPT == "true"
    text = prompt_text(body)
    is_long = len(text) > LONG_PROMPT_CHARS
    pool = "long" if is_long else "short"

    if not body.get("stream"):
        return await non_streaming(body, text, is_long, pool)

    return StreamingResponse(stream_with_retry(body, text, is_long, pool),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def stream_with_retry(body: dict, text: str, is_long: bool, pool: str):
    t0 = time.perf_counter()
    tried: set[str] = set()
    QUEUE.inc()
    try:
        for attempt in range(MAX_ATTEMPTS):
            backend = pick_backend(text, is_long, tried)
            if backend is None:
                break
            tried.add(backend)
            if attempt:
                RETRIES.inc()

            sent_any = False
            _track(backend, is_long, +1)
            try:
                stats = _StreamStats()
                async with client.stream("POST", f"http://{backend}/v1/chat/completions",
                                         json=body) as r:
                    if r.status_code != 200:
                        raise httpx.HTTPStatusError("bad status", request=r.request, response=r)
                    async for chunk in r.aiter_raw():
                        stats.observe(chunk, t0, pool)
                        sent_any = True
                        yield chunk  # pass bytes straight through: no buffering
                TOKENS.labels(pool).inc(stats.tokens())
                E2E.labels(pool).observe(time.perf_counter() - t0)
                REQUESTS.labels(pool, "ok").inc()
                return
            except (httpx.HTTPError, OSError) as exc:
                bad_until[backend] = time.monotonic() + COOLDOWN_S
                if sent_any:
                    # Tokens already reached the client; can't replay safely.
                    ERRORS.labels("midstream_" + type(exc).__name__).inc()
                    REQUESTS.labels(pool, "error").inc()
                    return
                ERRORS.labels("upstream_" + type(exc).__name__).inc()
            finally:
                _track(backend, is_long, -1)

        REQUESTS.labels(pool, "error").inc()
        ERRORS.labels("no_backend_succeeded").inc()
        yield b'data: {"error": "all backends failed"}\n\ndata: [DONE]\n\n'
    finally:
        QUEUE.dec()


async def non_streaming(body: dict, text: str, is_long: bool, pool: str):
    tried: set[str] = set()
    for _ in range(MAX_ATTEMPTS):
        backend = pick_backend(text, is_long, tried)
        if backend is None:
            break
        tried.add(backend)
        _track(backend, is_long, +1)
        try:
            r = await client.post(f"http://{backend}/v1/chat/completions", json=body)
            REQUESTS.labels(pool, "ok" if r.status_code == 200 else "error").inc()
            return JSONResponse(r.json(), status_code=r.status_code)
        except (httpx.HTTPError, OSError):
            bad_until[backend] = time.monotonic() + COOLDOWN_S
        finally:
            _track(backend, is_long, -1)
    ERRORS.labels("no_backend_succeeded").inc()
    return JSONResponse({"error": "all backends failed"}, status_code=503)


def _track(backend: str, is_long: bool, delta: int) -> None:
    inflight[backend] = inflight.get(backend, 0) + delta
    if is_long:
        inflight_long[backend] = inflight_long.get(backend, 0) + delta
    INFLIGHT.labels(backend).set(inflight[backend])


class _StreamStats:
    """Peeks at SSE lines as they pass through, without delaying them."""

    def __init__(self) -> None:
        self.buf = b""
        self.first: float | None = None
        self.last: float | None = None
        self.content_chunks = 0
        self.usage_tokens: int | None = None

    def observe(self, chunk: bytes, t0: float, pool: str) -> None:
        self.buf += chunk
        *lines, self.buf = self.buf.split(b"\n")
        for line in lines:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                continue
            try:
                ev = json.loads(data)
            except ValueError:
                continue
            usage = ev.get("usage")
            if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                self.usage_tokens = int(usage["completion_tokens"])
            choices = ev.get("choices") or []
            if choices and (choices[0].get("delta") or {}).get("content"):
                now = time.perf_counter()
                if self.first is None:
                    self.first = now
                    TTFT.labels(pool).observe(now - t0)
                elif self.last is not None:
                    ITL.observe(now - self.last)
                self.last = now
                self.content_chunks += 1

    def tokens(self) -> int:
        return self.usage_tokens if self.usage_tokens is not None else self.content_chunks

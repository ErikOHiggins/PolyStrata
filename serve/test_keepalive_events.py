"""The config's "keepalive_events" (opt-in): the stream's keep-alives are the API's own events instead of SSE comments.
An older openai-go (pool, poolside's agent, is built on one) ends the stream on the first comment, so a prompt that
reads for a while never gets its answer.  Chat Completions sends a chunk with an empty delta, Anthropic its own
`event: ping`; the Responses API has no keep-alive event and keeps the comment.  Off, nothing changes.

    python -m unittest serve.test_keepalive_events -v
"""
import contextlib
import http.client
import io
import json
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

import serve.server as server
from serve.frontend import ChatTemplate
from serve.server import (ByteTokenizer, MockEngine, Service, model_settings, openai_collect, serve,
                          structured_chunks)
from serve.structured import prepare_format

ROOT = Path(__file__).resolve().parents[1]
PINGS = 3                                         # the heartbeats Reading sends before it answers
ANSWER = "Hello there."


def old_sdk_events(raw: str) -> list:
    """The stream as an older openai-go (and anthropic-sdk-go, the same decoder) reads it: every blank line ends an
    event, also one after a line that was only a comment, and every event's data is parsed as JSON.  A comment's empty
    data raises here as it fails there ("unexpected end of JSON input"), which ends the client's stream."""
    events, kind, data = [], "", []
    for line in raw.splitlines():
        if not line:
            if "".join(data).startswith("[DONE]"):
                return events
            events.append((kind, json.loads("\n".join(data))))
            kind, data = "", []
            continue
        name, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if name == "event":
            kind = value
        elif name == "data":
            data.append(value)
    return events


class Reading(MockEngine):
    """Reads its prompt in PINGS steps, a heartbeat each (as the engine's PP lines reach the server), then answers."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        for _ in range(PINGS):
            yield None
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class HttpCase(unittest.TestCase):
    script = ANSWER

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = Reading(self.tok, self.script, max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with contextlib.redirect_stdout(io.StringIO()), urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
        return raw if r.headers.get("Content-Type") == "text/event-stream" else json.loads(raw)

    def chat(self, **req):
        return self.post("/v1/chat/completions", {"model": "m", "max_tokens": 64, "reasoning_effort": "none",
                                                  "messages": [{"role": "user", "content": "hi"}], **req})


class ChatCompletions(HttpCase):
    def test_off_by_default_the_keep_alive_is_the_comment(self):
        self.assertIs(self.svc.keepalive_events, False)
        raw = self.chat(stream=True)
        self.assertEqual(raw.count(": keep-alive\n\n"), PINGS)
        with self.assertRaises(json.JSONDecodeError):
            old_sdk_events(raw)                       # the failure this setting is for

    def test_on_every_keep_alive_is_a_chunk_with_an_empty_delta(self):
        self.svc.keepalive_events = True
        raw = self.chat(stream=True)
        self.assertNotIn(": keep-alive", raw)
        chunks = [c for _, c in old_sdk_events(raw)]
        beats = [c for c in chunks if c["choices"][0]["delta"] == {} and c["choices"][0]["finish_reason"] is None]
        self.assertEqual(len(beats), PINGS)
        self.assertEqual({(c["id"], c["object"], c["created"], c["model"]) for c in chunks},
                         {(chunks[0]["id"], "chat.completion.chunk", chunks[0]["created"], chunks[0]["model"])},
                         "a keep-alive is a chunk of the same completion")
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant", "content": ""}, "the role comes first")
        self.assertEqual("".join(c["choices"][0]["delta"].get("content") or "" for c in chunks), ANSWER)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertIn("usage", chunks[-1])

    def test_progress_still_rides_on_the_keep_alive(self):
        self.svc.keepalive_events = True
        self.engine.progress, self.engine.progress_ms = (8192, 25000), 2200.0
        raw = self.chat(stream=True, return_progress=True)
        self.assertNotIn(": keep-alive", raw)
        chunks = [c for _, c in old_sdk_events(raw)]
        self.assertEqual(sum("prompt_progress" in c for c in chunks), PINGS)

    def test_a_non_streamed_answer_is_the_same(self):
        off = self.chat()
        self.svc.keepalive_events = True
        on = self.chat()
        for r in (off, on):
            del r["id"], r["created"]
            r.pop("timings", None)
        self.assertEqual(on, off)

    def test_the_monitor_keeps_the_answer_alone(self):
        self.svc.api_monitor = True
        self.svc.keepalive_events = True
        self.chat(stream=True)
        record = self.svc.api_requests[-1]
        self.assertEqual(record["output"], ANSWER)
        self.assertEqual(record["usage"]["completion_tokens"], len(self.tok.encode(ANSWER)) + 1)

    def test_a_queued_request_gets_chunks_while_it_waits(self):
        # #1619's queue beat comes through the same ping
        self.svc.keepalive_events = True
        self.engine = self.svc.engine = MockEngine(self.tok, ANSWER, max_context=16384)
        with mock.patch.object(server, "QUEUE_BEAT_S", 0.1):
            self.svc.fifo.acquire()                   # another request holds the turn
            c = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=10)
            self.addCleanup(c.close)
            body = {"stream": True, "reasoning_effort": "none", "messages": [{"role": "user", "content": "hi"}]}
            c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            seen, t0 = b"", time.monotonic()
            while seen.count(b'"delta": {}') < 2 and time.monotonic() - t0 < 5:
                seen += r.fp.readline()
            self.svc.fifo.release()
            raw = seen.decode() + r.read().decode()
        self.assertGreaterEqual(raw.count('"delta": {}, "finish_reason": null'), 2, raw)
        self.assertNotIn(": keep-alive", raw)
        self.assertEqual("".join(c["choices"][0]["delta"].get("content") or "" for _, c in old_sdk_events(raw)),
                         ANSWER)


class Structured(HttpCase):
    """A structured answer is held until it is validated; the keep-alives sent meanwhile follow the setting."""

    script = '{"answer": 4}'
    FORMAT = {"type": "json_object"}

    def test_on_the_held_answer_is_kept_alive_with_chunks(self):
        self.svc.keepalive_events = True
        raw = self.chat(stream=True, response_format=self.FORMAT)
        self.assertNotIn(": keep-alive", raw)
        chunks = [c for _, c in old_sdk_events(raw)]
        self.assertGreaterEqual(sum(c["choices"][0]["delta"] == {} and c["choices"][0]["finish_reason"] is None
                                    for c in chunks), PINGS)
        content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
        self.assertEqual(json.loads(content), {"answer": 4})
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_off_it_is_the_comment_as_before(self):
        raw = self.chat(stream=True, response_format=self.FORMAT)
        self.assertGreaterEqual(raw.count(": keep-alive\n\n"), PINGS)

    def chunks(self):
        def chunk(delta, finish=None):
            return {"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 1, "model": "m",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        yield chunk({"role": "assistant", "content": ""})
        for piece in ('{"answer"', ': 4', '}'):
            yield chunk({"content": piece})
        yield {**chunk({}, "stop"), "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4}}

    def test_its_own_beat_while_the_answer_is_generated(self):
        # every chunk comes a second after the last beat: each of the five sends a keep-alive in its place
        _, validator = prepare_format(self.FORMAT, [{"role": "user", "content": "hi"}])
        clock = iter(range(0, 1000, 2))
        with mock.patch.object(server.time, "monotonic", lambda: next(clock)):
            off = list(structured_chunks(self.chunks(), validator))
            on = list(structured_chunks(self.chunks(), validator, events=True))
        self.assertEqual(off[:-3], [None] * 5)
        self.assertEqual(on[:-3], [{"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 1, "model": "m",
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": None}]}] * 5)
        self.assertEqual(on[-3:], off[-3:], "the validated answer is the same")
        self.assertEqual(json.loads(on[-2]["choices"][0]["delta"]["content"]), {"answer": 4})


class Anthropic(HttpCase):
    def setUp(self):
        super().setUp()
        self.svc.anthropic_think_unasked = False      # "anthropic_thinking": "on_request": the script is the answer

    def messages(self, **req):
        return self.post("/v1/messages", {"model": "m", "max_tokens": 64,
                                          "messages": [{"role": "user", "content": "hi"}], **req})

    def test_off_by_default_the_keep_alive_is_the_comment(self):
        raw = self.messages(stream=True)
        self.assertEqual(raw.count(": keep-alive\n\n"), PINGS)
        self.assertNotIn("event: ping", raw)

    def test_on_it_is_anthropics_own_ping(self):
        self.svc.keepalive_events = True
        raw = self.messages(stream=True)
        self.assertNotIn(": keep-alive", raw)
        self.assertEqual(raw.count('event: ping\ndata: {"type": "ping"}\n\n'), PINGS)
        events = old_sdk_events(raw)
        self.assertEqual([k for k, _ in events][:1 + PINGS], ["message_start"] + ["ping"] * PINGS)
        self.assertTrue(all(k == e["type"] for k, e in events))
        self.assertEqual(events[-1][0], "message_stop")
        self.assertEqual("".join(e["delta"].get("text", "") for k, e in events if k == "content_block_delta"), ANSWER)

    def test_a_non_streamed_answer_is_the_same(self):
        off = self.messages()
        self.svc.keepalive_events = True
        on = self.messages()
        del off["id"], on["id"]
        self.assertEqual(on, off)


class Responses(HttpCase):
    def test_the_responses_api_keeps_the_comment(self):
        # it has no keep-alive event; Codex gets a response.in_progress every 15 s as before
        self.svc.keepalive_events = True
        raw = self.post("/v1/responses", {"model": "m", "input": "hi", "stream": True})
        self.assertEqual(raw.count(": keep-alive\n\n"), PINGS)


class Setting(unittest.TestCase):
    def test_the_run_configs_key(self):
        self.assertIs(model_settings({})["keepalive_events"], False)
        self.assertIs(model_settings({"keepalive_events": False})["keepalive_events"], False)
        self.assertIs(model_settings({"keepalive_events": True})["keepalive_events"], True)
        with self.assertRaisesRegex(ValueError, "keepalive_events"):
            model_settings({"keepalive_events": "yes"})


if __name__ == "__main__":
    unittest.main()

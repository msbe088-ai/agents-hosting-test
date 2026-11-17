#!/usr/bin/env python3
"""Tiny hosting-test agent: fetch AAPL + BTC, optionally run one GGUF generation, write static files."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

AAPL_TICKER = "AAPL"
BINANCE_URL = "https://data-api.binance.vision/api/v3/ticker/price?symbol=BTCUSDT"
GGUF_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
GGUF_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
PROMPT = "Reply with the single word OK."


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except OSError:
        return None
    return None


def fetch_aapl() -> dict[str, Any]:
    started = time.perf_counter()
    out: dict[str, Any] = {"source": "yfinance", "symbol": AAPL_TICKER, "ok": False}
    try:
        import yfinance as yf

        ticker = yf.Ticker(AAPL_TICKER)
        hist = ticker.history(period="5d", interval="1d")
        if hist is None or hist.empty:
            out["error"] = "empty history"
        else:
            last = hist.iloc[-1]
            out.update(
                {
                    "ok": True,
                    "last_date": str(hist.index[-1].date()) if hasattr(hist.index[-1], "date") else str(hist.index[-1]),
                    "close": float(last["Close"]),
                    "open": float(last["Open"]),
                    "high": float(last["High"]),
                    "low": float(last["Low"]),
                    "volume": float(last["Volume"]),
                    "rows": int(len(hist)),
                }
            )
    except Exception as exc:  # noqa: BLE001 — test harness must record any failure
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()
    out["seconds"] = round(time.perf_counter() - started, 3)
    return out


def fetch_btc() -> dict[str, Any]:
    started = time.perf_counter()
    out: dict[str, Any] = {"source": "data-api.binance.vision", "symbol": "BTCUSDT", "ok": False, "url": BINANCE_URL}
    try:
        req = Request(BINANCE_URL, headers={"User-Agent": "agents-hosting-test/1.0"})
        with urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            out["http_status"] = int(resp.status)
        data = json.loads(body)
        price = float(data["price"])
        out.update({"ok": True, "price": price, "raw": data})
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()
    out["seconds"] = round(time.perf_counter() - started, 3)
    return out


def run_llm(outdir: Path) -> dict[str, Any]:
    started_all = time.perf_counter()
    out: dict[str, Any] = {
        "ok": False,
        "repo": GGUF_REPO,
        "filename": GGUF_FILE,
        "prompt": PROMPT,
        "rss_before_mb": rss_mb(),
    }
    try:
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        t0 = time.perf_counter()
        path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)
        out["download_seconds"] = round(time.perf_counter() - t0, 3)
        out["path"] = path
        out["size_mb"] = round(os.path.getsize(path) / (1024 * 1024), 1)

        t1 = time.perf_counter()
        n_threads = min(4, os.cpu_count() or 2)
        llm = Llama(model_path=path, n_ctx=512, n_threads=n_threads, verbose=False)
        out["load_seconds"] = round(time.perf_counter() - t1, 3)
        out["n_threads"] = n_threads
        out["rss_after_load_mb"] = rss_mb()

        t2 = time.perf_counter()
        result = llm.create_chat_completion(
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=16,
            temperature=0.0,
        )
        gen_s = time.perf_counter() - t2
        out["generation_seconds"] = round(gen_s, 3)
        choice = (result.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or choice.get("text") or ""
        out["answer"] = text
        usage = result.get("usage") or {}
        out["usage"] = usage
        completion = usage.get("completion_tokens")
        if completion and gen_s > 0:
            out["tokens_per_sec"] = round(completion / gen_s, 3)
        out["rss_after_gen_mb"] = rss_mb()
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()
    out["total_seconds"] = round(time.perf_counter() - started_all, 3)
    (outdir / "llm.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_existing(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def write_html(data: dict[str, Any], out_path: Path) -> None:
    aapl = data.get("aapl") or {}
    btc = data.get("btc") or {}
    llm = data.get("llm")
    generated = data.get("generated_utc") or ""
    aapl_line = (
        f"AAPL close {aapl.get('close')} on {aapl.get('last_date')} (yfinance, {aapl.get('seconds')}s)"
        if aapl.get("ok")
        else f"AAPL failed: {aapl.get('error')}"
    )
    btc_line = (
        f"BTCUSDT {btc.get('price')} (data-api.binance.vision, {btc.get('seconds')}s)"
        if btc.get("ok")
        else f"BTC failed: {btc.get('error')}"
    )
    if llm is None:
        llm_line = "LLM test was not run in this job."
    elif llm.get("ok"):
        llm_line = (
            f"Qwen2.5-1.5B Q4_K_M OK. download {llm.get('download_seconds')}s, "
            f"load {llm.get('load_seconds')}s, generation {llm.get('generation_seconds')}s, "
            f"tokens/s {llm.get('tokens_per_sec')}, answer={llm.get('answer')!r}"
        )
    else:
        llm_line = f"LLM test failed: {llm.get('error')}"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>agents-hosting-test</title>
  <style>
    body {{ font-family: sans-serif; max-width: 52rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.45; }}
    pre {{ background: #f4f4f4; padding: 1rem; overflow: auto; }}
    .ok {{ color: #0a7a2f; }}
    .fail {{ color: #b00020; }}
  </style>
</head>
<body>
  <h1>agents-hosting-test</h1>
  <p>One-shot GitHub Actions + GitHub Pages hosting probe. Generated at {generated} (UTC).</p>
  <ul>
    <li class="{'ok' if aapl.get('ok') else 'fail'}">{aapl_line}</li>
    <li class="{'ok' if btc.get('ok') else 'fail'}">{btc_line}</li>
    <li>{llm_line}</li>
  </ul>
  <p>Raw JSON: <a href="data.json">data.json</a></p>
  <pre>{json.dumps(data, ensure_ascii=False, indent=2)}</pre>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="site")
    parser.add_argument("--llm-only", action="store_true")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_path = outdir / "data.json"

    if args.llm_only:
        data = load_existing(data_path)
        data["llm"] = run_llm(outdir)
        data["generated_utc"] = now_iso()
        data["host"] = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpus": os.cpu_count(),
        }
        data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        write_html(data, outdir / "index.html")
        print(json.dumps({"llm_ok": data["llm"].get("ok"), "error": data["llm"].get("error")}, ensure_ascii=False))
        return 0 if data["llm"].get("ok") else 1

    data: dict[str, Any] = {
        "generated_utc": now_iso(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpus": os.cpu_count(),
        },
        "aapl": fetch_aapl(),
        "btc": fetch_btc(),
    }
    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(data, outdir / "index.html")
    print(json.dumps({"aapl_ok": data["aapl"]["ok"], "btc_ok": data["btc"]["ok"]}, ensure_ascii=False))
    return 0 if data["aapl"]["ok"] and data["btc"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

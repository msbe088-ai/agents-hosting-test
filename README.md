# agents-hosting-test

One-shot public test of GitHub Pages + GitHub Actions as a free host for a planned finance agents page.

- Trigger: `workflow_dispatch` only. No cron. Nothing keeps running after this test.
- The workflow fetches AAPL via yfinance and BTC via `data-api.binance.vision`, writes static HTML/JSON, and deploys to GitHub Pages.
- An optional input runs one short Qwen2.5-1.5B Instruct Q4_K_M GGUF generation on the runner and records timing. The model is downloaded at job time and is not stored in this repository.

This repository is a hosting probe, not the product.

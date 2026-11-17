# ThreatWeigh demo video

`threatweigh_demo.mp4` (1280×720, ~3 min 10 s, no audio; on-screen captions explain each step).

| Time | Scene | Source |
|---|---|---|
| 0:00 | Title | `deck.html` |
| 0:06 | The problem: CVE text → 8 metrics → 9.8 Critical | `deck.html` |
| 0:17 | Data: 137,521 CVEs, label leakage removed, de-duplication, splits | `deck.html` |
| 0:29 | Model flow: SBERT + expert lexicon + KeyBERT → symbolic weight → fused pooling | `deck.html` |
| 0:46 | Real token weights (11 / 2 / 1) for one sentence, AV model | computed with `src/model.symbolic_weights` |
| 0:59 | CLI: `python predict.py --explain …`, **real output** (`cli_output.txt`) | replayed in `deck.html` |
| 1:19 | **Live Streamlit app**: RCE example, 8 predictions, IG word highlights | Playwright screen recording |
| 1:47 | Live app: stored-XSS example and an honest failure ("administrator" → PR:High) | Playwright screen recording |
| 2:17 | Live app: Model results tab, How it works tab | Playwright screen recording |
| 2:52 | Results summary, then the commands to run everything | `deck.html` |

Integrated Gradients for 8 models takes ~30 s per prediction on a CPU. Those two waits are sped up 8×,
and an on-screen label says so. Everything else plays in real time.

## Re-create it

```bash
python predict.py --explain "A race condition in the kernel driver allows local users to gain root privileges via a crafted ioctl call." > video/cli_output.txt
```

```bash
(cd video && python -m http.server 8600) &
```

```bash
streamlit run app.py --server.headless true --server.port 8599 &
```

```bash
node video/record.js video/raw
```

```bash
python video/build_video.py
```

Requires trained models in `models/` (`bash scripts/run_all.sh`), Node with Playwright, and ffmpeg.
- [youtube/](youtube/) – YouTube upload template: titles, description with chapters, tags, thumbnail.

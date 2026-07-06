# Demo screenshots

Place UTF-8-friendly PNG/WebP screenshots here for README and portfolio.

Suggested filenames:

| File | Content |
|------|---------|
| `demo-cli-report.png` | Terminal running `run_demo.py` with Chinese report excerpt |
| `demo-web-upload.png` | Web UI at `http://127.0.0.1:8000` with PDF upload |
| `demo-audit-timeline.png` | Agent timeline / chart from generated report |
| `demo-state-rag.png` | `*_state.json` showing `rag_evidence` citations |

Capture tips (Windows):

1. Run `.\scripts\ensure_utf8.ps1` before demo so terminal Chinese is readable.
2. Use `Win + Shift + S` region capture, save as PNG.
3. Keep images under 1920px width; GitHub renders well at ~1200px.

After adding images, reference them from root `README.md`:

```markdown
![CLI demo](docs/assets/demo-cli-report.png)
```

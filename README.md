# LangChain OpenSCAD Render Agent

This project provides a Python LangChain agent that:

1. Uses a single tool: `render_openscad`
2. Calls the `openscad` binary to render SCAD code into one or more PNG images
3. Applies deterministic OpenSCAD overlays on screenshots (`axes`, `scales`, `edges`)
4. Computes mesh statistics and a universal edge count from exported STL
5. Loads API credentials from `.env` at runtime
6. Defaults to annotation-friendly high-resolution renders (`1600x1200`)

It also includes a standalone MCP server (`mcp_server.py`) so any MCP-compatible
LLM client can call OpenSCAD rendering as a tool and receive image content blocks.

It also includes a rich chat-like TUI (`tui.py`) for iterative image-to-OpenSCAD reconstruction.

## Files

- `agent.py` - main agent entrypoint
- `tui.py` - rich chat-like iterative reconstruction TUI
- `mcp_server.py` - standalone MCP tool server over stdio
- `mcp_http_server.py` - standalone MCP server over HTTP (`/mcp`)
- `mcp_servers.example.json` - sample MCP client server config
- `.env` - runtime environment variables (includes dummy key)
- `requirements.txt` - Python dependencies

## Prerequisites

- Python 3.10+
- OpenSCAD installed and available on your PATH as `openscad`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

`.env` is already included with a dummy key:

```env
OPENAI_API_KEY=dummy-openai-key
OPENAI_MODEL=gpt-5
```

Replace `OPENAI_API_KEY` with a real value before real API calls.

## Run

CLI agent:

```bash
python agent.py "Create a 20mm cube and render shots at 512x512, 1024x768, and 1600x1200, then describe all images."
```

Interactive CLI mode:

```bash
python agent.py
```

Rich TUI mode:

```bash
python tui.py
```

MCP server mode (stdio transport):

```bash
python mcp_server.py
```

MCP server mode (HTTP transport):

```bash
python mcp_http_server.py --host 127.0.0.1 --port 8765
```

### MCP tool contract

The server exposes one MCP tool: `render_openscad`.

- Input:
  - `scad_code` (string, required)
  - `image_sizes` (array of `WIDTHxHEIGHT` strings, optional)
- Output:
  - `content` blocks including:
    - `type: "text"` status/details
    - `type: "image"` with base64 `data` and `mimeType: "image/png"`
  - `structuredContent` with overlay metadata, mesh stats, edge counter, and image paths

Example MCP client config is provided in `mcp_servers.example.json`.

### TUI workflow

- Paste image from clipboard (`Ctrl+V`, fallback `Ctrl+Shift+V` / `Shift+Insert`) or press `Paste Clipboard Image`
- Each attached image gets a tag like `[Image 1]`
- Mention tags in prompt text to pass those images into multimodal model context
- Tags behave atomically in the prompt: one Backspace after a tag removes the whole tag
- Removing a tag from the prompt also deletes that attachment from the active attachment list
- Interrupt controls: `Esc` twice quickly or `Ctrl+C` to stop a running loop; repeat to exit TUI
- The app auto-iterates: propose SCAD -> render via tool -> visually review -> revise until stop
- Every iteration auto-attaches fixed labeled shots (model has zero control):
- `iso with top + NORTH EAST seen`, `zoomed x3 iso with top + NORTH EAST seen`, `iso with top+SOUTH WEST`, `iso with BOTTOM NORTH EAST`, `iso with BOTTOM SOUTH WEST`
- Both designer and reviewer keep persistent run memory (all prior code, generated images, and ratings)
- Tool parser/runtime failures trigger same-iteration `syntax retry`: error is fed back to generator until valid render or retry cap
- Iterations, tool calls, review summaries, and output paths are shown in chat panels

### TUI inline commands

- `/attach /path/to/image.png` attach image and auto-insert a tag
- `/remove 2` remove `[Image 2]`

Renders are saved under `renders/`.
Reference images pasted from clipboard are saved under `reference_images/`.
Final SCAD outputs from TUI are saved under `outputs/`.
Full timestamped JSON session logs are saved under `session_logs/` (one file per TUI launch).

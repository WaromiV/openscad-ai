import base64
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO, cast

from agent import RENDERS_DIR, render_openscad


SERVER_NAME = "openscad-render-mcp"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2025-06-18"

TOOL_NAME = "render_openscad"
TOOL_DESCRIPTION = (
    "Render OpenSCAD source code into 6 deterministic PNG views. "
    "Returns MCP image content blocks with base64 payloads and fixed XYZ color legend."
    "The 6 views are: top_ne, top_sw, bottom_ne, bottom_sw, top, and a zoomed x3 version of top_ne. "
    "OpenSCAD facet controls (the practical FN settings to use) are: "
    "$fn: fixed number of fragments (explicit polygon sides; great for final quality cylinders/spheres), "
    "$fa: minimum angle per fragment in degrees (adaptive smoothness based on curvature; good global quality control), "
    "$fs: minimum fragment size in model units (adaptive smoothness based on physical segment length; good for scale-consistent tessellation). "
    "How they interact: if $fn is set > 0 it overrides $fa/$fs; if $fn is 0 or unset then OpenSCAD derives fragments from $fa and $fs together. "
    "Common usage guidance: quick preview uses lower detail (example $fn=24 or coarse $fa/$fs), final export uses higher detail (example $fn=96+, or tighter $fa/$fs such as $fa=4 and $fs=0.5). "
    "For threaded, press-fit, and screw interfaces, increase tessellation to reduce fit error from faceting. "
    "When integrating some parts into you design like bits or screws, you must always think about how they will fit or integrate with the rest of the design. You should always consider the dimensions, tolerances, and how the parts will be assembled together including convergence of holes alowing for screws or press fit, and how the parts will interact with each other in the final design. Always keep in mind the practical aspects of manufacturing and assembly when designing with OpenSCAD or any CAD software. If we are planning to make a hole, check if it doesn't pass through another hole. Mind the fn= parameter."
)

BASE_SHOTS: tuple[tuple[str, str, tuple[int, int, int]], ...] = (
    ("top_ne", "iso with top + NORTH EAST seen", (55, 0, 45)),
    ("top_sw", "iso with top+SOUTH WEST", (55, 0, 225)),
    ("bottom_ne", "iso with BOTTOM NORTH EAST", (-55, 0, 45)),
    ("bottom_sw", "iso with BOTTOM SOUTH WEST", (-55, 0, 225)),
    ("top", "top view", (0, 0, 0)),
)
ZOOM_SHOT_LABEL = "zoomed x3 iso with top + NORTH EAST seen"


class MCPError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _log(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def _read_message(stdin: TextIO) -> dict[str, Any] | None:
    headers: dict[str, str] = {}

    while True:
        line = stdin.readline()
        if line == "":
            return None
        if line in {"\r\n", "\n"}:
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers[key.strip().lower()] = value.strip()

    content_length_raw = headers.get("content-length")
    if content_length_raw is None:
        raise MCPError(-32600, "Missing Content-Length header")

    try:
        content_length = int(content_length_raw)
    except ValueError as exc:
        raise MCPError(-32600, "Invalid Content-Length header") from exc

    body = stdin.read(content_length)
    if body == "":
        return None

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise MCPError(-32700, f"Invalid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise MCPError(-32600, "Request body must be a JSON object")
    return parsed


def _write_message(stdout: TextIO, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    data = encoded.encode("utf-8")
    stdout.write(f"Content-Length: {len(data)}\r\n\r\n")
    stdout.write(encoded)
    stdout.flush()


def _ok_response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _tool_schema() -> dict[str, Any]:
    return {
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "inputSchema": {
            "type": "object",
            "properties": {
                "scad_code": {
                    "type": "string",
                    "description": "OpenSCAD source code to render.",
                },
                "image_sizes": {
                    "type": "array",
                    "description": "Deprecated. First entry is used as fixed_render_size.",
                    "items": {"type": "string"},
                    "default": ["1600x1200"],
                },
                "fixed_render_size": {
                    "type": "string",
                    "description": "Fixed size for all 6 views (WIDTHxHEIGHT).",
                    "default": "1600x1200",
                },
            },
            "required": ["scad_code"],
            "additionalProperties": False,
        },
    }


def _view_scad(
    scad_code: str, vpr: tuple[int, int, int], vpt: tuple[float, float, float]
) -> str:
    rx, ry, rz = vpr
    cx, cy, cz = vpt
    return f"$vpr=[{rx},{ry},{rz}];\n$vpt=[{cx},{cy},{cz}];\n{scad_code}\n"


def _load_font(font_size: int):
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), font_size)
    return ImageFont.load_default()


def _bake_top_label(image_path: Path, label_text: str) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGBA")
    width, height = image.size
    banner_height = max(46, int(height * 0.09))

    overlay = Image.new("RGBA", (width, banner_height), (0, 0, 0, 200))
    image.alpha_composite(overlay, (0, 0))

    draw = ImageDraw.Draw(image)
    font_size = max(18, int(height * 0.032))
    font = _load_font(font_size)

    left, top, right, bottom = draw.textbbox(
        (0, 0),
        label_text,
        font=font,
        stroke_width=2,
    )
    text_width = right - left
    text_height = bottom - top
    x = max(12, (width - text_width) // 2)
    y = max(6, (banner_height - text_height) // 2)

    draw.text(
        (x, y),
        label_text,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 255),
    )

    image.convert("RGB").save(image_path, format="PNG")


def _bake_xyz_legend(image_path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGBA")
    width, height = image.size
    pad = max(8, int(min(width, height) * 0.02))
    box_w = max(150, int(width * 0.18))
    box_h = max(72, int(height * 0.12))

    overlay = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 170))
    image.alpha_composite(overlay, (pad, pad))
    draw = ImageDraw.Draw(image)

    font_size = max(14, int(height * 0.028))
    font = _load_font(font_size)

    x0 = pad + 14
    y0 = pad + 16
    spacing = max(18, int(box_h * 0.28))
    line_len = max(22, int(box_w * 0.24))

    legend = [
        ("X", (235, 64, 52)),
        ("Y", (60, 179, 113)),
        ("Z", (65, 105, 225)),
    ]

    for index, (label, color) in enumerate(legend):
        y = y0 + index * spacing
        draw.line(
            (x0, y, x0 + line_len, y),
            fill=(*color, 255),
            width=max(2, int(font_size * 0.16)),
        )
        draw.text(
            (x0 + line_len + 10, y - int(font_size * 0.45)),
            label,
            font=font,
            fill=(*color, 255),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 255),
        )

    image.convert("RGB").save(image_path, format="PNG")


def _create_zoomed_image(source_path: Path, zoom_factor: int, label_text: str) -> Path:
    from PIL import Image

    image = Image.open(source_path).convert("RGB")
    width, height = image.size

    crop_width = max(2, width // zoom_factor)
    crop_height = max(2, height // zoom_factor)
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    right = min(width, left + crop_width)
    bottom = min(height, top + crop_height)

    crop = image.crop((left, top, right, bottom))
    zoomed = crop.resize((width, height), Image.Resampling.LANCZOS)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    output = RENDERS_DIR / f"render_zoomx{zoom_factor}_{timestamp}.png"
    zoomed.save(output, format="PNG")
    _bake_top_label(output, label_text)
    _bake_xyz_legend(output)
    return output


def _view_center(scad_code: str, fixed_render_size: str) -> tuple[float, float, float]:
    try:
        tool_output = cast(Any, render_openscad).invoke(
            {"scad_code": scad_code, "image_sizes": [fixed_render_size]}
        )
        payload = json.loads(tool_output)
    except Exception as exc:
        raise RuntimeError(f"Failed to compute mesh center: {exc}")

    if not isinstance(payload, dict):
        return (0.0, 0.0, 0.0)
    mesh_stats = payload.get("mesh_stats", {})
    if not isinstance(mesh_stats, dict):
        return (0.0, 0.0, 0.0)

    bbox_min = mesh_stats.get("bbox_min")
    bbox_max = mesh_stats.get("bbox_max")
    if not isinstance(bbox_min, list) or not isinstance(bbox_max, list):
        return (0.0, 0.0, 0.0)
    if len(bbox_min) != 3 or len(bbox_max) != 3:
        return (0.0, 0.0, 0.0)

    try:
        return (
            round((float(bbox_min[0]) + float(bbox_max[0])) / 2.0, 6),
            round((float(bbox_min[1]) + float(bbox_max[1])) / 2.0, 6),
            round((float(bbox_min[2]) + float(bbox_max[2])) / 2.0, 6),
        )
    except Exception:
        return (0.0, 0.0, 0.0)


def _render_fixed_shots(scad_code: str, fixed_render_size: str) -> dict[str, Any]:
    shot_outputs: dict[str, dict[str, Any]] = {}
    first_payload: dict[str, Any] | None = None
    center = _view_center(scad_code, fixed_render_size)

    for shot_key, shot_label, shot_vpr in BASE_SHOTS:
        oriented_scad = _view_scad(scad_code, shot_vpr, center)
        try:
            tool_output = cast(Any, render_openscad).invoke(
                {"scad_code": oriented_scad, "image_sizes": [fixed_render_size]}
            )
        except Exception as exc:
            raise RuntimeError(f"Shot '{shot_label}' failed: {exc}")

        payload = json.loads(tool_output)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Shot '{shot_label}' returned non-dict payload")
        renders = payload.get("renders")
        if not isinstance(renders, list) or not renders:
            raise RuntimeError(f"Shot '{shot_label}' returned no renders")

        render = renders[0]
        if not isinstance(render, dict):
            raise RuntimeError(f"Shot '{shot_label}' render entry malformed")

        image_path = Path(str(render.get("image_path", "")))
        if not image_path.exists():
            raise RuntimeError(f"Shot '{shot_label}' image missing: {image_path}")

        _bake_top_label(image_path, shot_label)
        _bake_xyz_legend(image_path)
        shot_outputs[shot_key] = {
            "image_path": str(image_path),
            "image_size": str(render.get("image_size", fixed_render_size)),
            "mime_type": str(render.get("mime_type", "image/png")),
            "openscad_log": str(render.get("openscad_log", "")),
            "view_key": shot_key,
            "view_label": shot_label,
            "zoom_factor": 1,
        }

        if first_payload is None:
            first_payload = payload

    top_ne = shot_outputs.get("top_ne")
    if top_ne is None:
        raise RuntimeError("Fixed shot 'top_ne' missing")

    top_ne_path = Path(str(top_ne["image_path"]))
    zoom_path = _create_zoomed_image(top_ne_path, 3, ZOOM_SHOT_LABEL)
    zoom_render = {
        "image_path": str(zoom_path),
        "image_size": str(top_ne.get("image_size", fixed_render_size)),
        "mime_type": "image/png",
        "openscad_log": "derived zoom from top_ne",
        "view_key": "top_ne_zoom_x3",
        "view_label": ZOOM_SHOT_LABEL,
        "zoom_factor": 3,
    }

    ordered_renders = [
        shot_outputs["top_ne"],
        zoom_render,
        shot_outputs["top_sw"],
        shot_outputs["bottom_ne"],
        shot_outputs["bottom_sw"],
        shot_outputs["top"],
    ]

    first_payload = first_payload or {}
    return {
        "renders": ordered_renders,
        "annotation_mode": "deterministic_cli_overlays_with_fixed_view_labels",
        "overlays": ["axes", "scales", "edges", "top_label"],
        "edge_counter": first_payload.get("edge_counter"),
        "mesh_stats": first_payload.get("mesh_stats"),
        "shot_policy": "fixed_6_shot_manifest_v2",
        "shot_manifest": [render["view_label"] for render in ordered_renders],
    }


def _run_render(arguments: dict[str, Any]) -> dict[str, Any]:
    scad_code = arguments.get("scad_code")
    if not isinstance(scad_code, str) or not scad_code.strip():
        raise MCPError(-32602, "'scad_code' must be a non-empty string")

    fixed_render_size = arguments.get("fixed_render_size", "1600x1200")
    if not isinstance(fixed_render_size, str) or not fixed_render_size.strip():
        raise MCPError(-32602, "'fixed_render_size' must be a non-empty string")

    image_sizes_arg = arguments.get("image_sizes")
    if image_sizes_arg is not None:
        if not isinstance(image_sizes_arg, list) or not all(
            isinstance(x, str) for x in image_sizes_arg
        ):
            raise MCPError(-32602, "'image_sizes' must be an array of strings")
        if image_sizes_arg:
            fixed_render_size = cast(str, image_sizes_arg[0])

    try:
        payload = _render_fixed_shots(scad_code, fixed_render_size)
    except Exception as exc:
        raise MCPError(-32603, f"OpenSCAD render failed: {exc}") from exc

    if not isinstance(payload, dict):
        raise MCPError(-32603, "Renderer payload must be a JSON object")

    renders = payload.get("renders")
    if not isinstance(renders, list) or not renders:
        raise MCPError(-32603, "Renderer payload missing 'renders'")
    if len(renders) != 6:
        raise MCPError(-32603, f"Renderer returned {len(renders)} images; expected 6")

    content: list[dict[str, Any]] = []
    image_info: list[dict[str, Any]] = []

    content.append(
        {
            "type": "text",
            "text": (
                "OpenSCAD render completed. "
                f"images={len(renders)}, "
                f"edge_counter={payload.get('edge_counter', 'unknown')}"
            ),
        }
    )

    for index, render in enumerate(renders, start=1):
        if not isinstance(render, dict):
            continue
        image_path = Path(str(render.get("image_path", "")))
        if not image_path.exists():
            raise MCPError(-32603, f"Rendered image missing on disk: {image_path}")

        image_bytes = image_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        mime_type = str(render.get("mime_type", "image/png"))
        image_size = str(render.get("image_size", "unknown"))

        image_info.append(
            {
                "index": index,
                "image_path": str(image_path),
                "image_size": image_size,
                "mime_type": mime_type,
                "view_label": str(render.get("view_label", "")),
            }
        )

        content.append(
            {
                "type": "text",
                "text": (
                    f"Image {index}: view={render.get('view_label', 'unknown')}, path={image_path}, size={image_size}, mime={mime_type}"
                ),
            }
        )
        content.append(
            {
                "type": "image",
                "data": image_b64,
                "mimeType": mime_type,
            }
        )

    structured = {
        "annotation_mode": payload.get("annotation_mode"),
        "overlays": payload.get("overlays", []),
        "mesh_stats": payload.get("mesh_stats", {}),
        "edge_counter": payload.get("edge_counter"),
        "shot_policy": payload.get("shot_policy"),
        "shot_manifest": payload.get("shot_manifest", []),
        "images": image_info,
    }

    return {
        "content": content,
        "structuredContent": structured,
        "isError": False,
    }


def _handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    if request.get("jsonrpc") != "2.0":
        raise MCPError(-32600, "Only JSON-RPC 2.0 requests are supported")

    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params", {})

    if not isinstance(method, str):
        raise MCPError(-32600, "Request method must be a string")

    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise MCPError(-32602, "params must be a JSON object")

    if method == "notifications/initialized":
        return None

    if request_id is None:
        return None

    if method == "initialize":
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
            "instructions": (
                "Use the render_openscad tool. Input OpenSCAD code, receive text + image content blocks."
            ),
        }
        return _ok_response(request_id, result)

    if method == "ping":
        return _ok_response(request_id, {})

    if method == "tools/list":
        return _ok_response(request_id, {"tools": [_tool_schema()]})

    if method == "tools/call":
        name = params.get("name")
        if name != TOOL_NAME:
            raise MCPError(-32602, f"Unknown tool: {name!r}")
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            raise MCPError(-32602, "'arguments' must be a JSON object")
        return _ok_response(request_id, _run_render(arguments))

    raise MCPError(-32601, f"Method not found: {method}")


def main() -> None:
    _log(f"Starting {SERVER_NAME} v{SERVER_VERSION} on stdio")

    while True:
        request_id: Any = None
        try:
            request = _read_message(sys.stdin)
            if request is None:
                break
            request_id = request.get("id")
            response = _handle_request(request)
            if response is not None:
                _write_message(sys.stdout, response)
        except MCPError as exc:
            if request_id is not None:
                _write_message(
                    sys.stdout, _error_response(request_id, exc.code, exc.message)
                )
            else:
                _log(f"Protocol error without request id: {exc.message}")
        except Exception as exc:
            if request_id is not None:
                _write_message(
                    sys.stdout,
                    _error_response(
                        request_id, -32603, f"Internal server error: {exc}"
                    ),
                )
            else:
                _log(f"Unhandled server error: {exc}")


if __name__ == "__main__":
    main()

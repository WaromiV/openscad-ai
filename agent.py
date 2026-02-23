import argparse
import base64
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
RENDERS_DIR = ROOT / "renders"


class RenderInput(BaseModel):
    scad_code: str = Field(..., description="OpenSCAD source code to render.")
    image_sizes: list[str] = Field(
        default_factory=lambda: ["1600x1200"],
        description=(
            "One or more image sizes formatted as WIDTHxHEIGHT, "
            "e.g. ['1024x768', '1600x1200']"
        ),
    )


def _validate_size(raw_size: str) -> tuple[int, int]:
    match = re.search(r"(\d+)\s*[x×]\s*(\d+)", raw_size)
    if match is None:
        raise ValueError(f"image_size must include WIDTHxHEIGHT, got: {raw_size!r}")
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return width, height


def _quantize_vertex(
    v: tuple[float, float, float], places: int = 6
) -> tuple[float, float, float]:
    return (round(v[0], places), round(v[1], places), round(v[2], places))


def _mesh_stats_from_ascii_stl(stl_path: Path) -> dict[str, Any]:
    vertex_re = re.compile(
        r"^\s*vertex\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )

    vertices: list[tuple[float, float, float]] = []
    triangles: list[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ] = []

    current: list[tuple[float, float, float]] = []
    with stl_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = vertex_re.match(line)
            if not match:
                continue
            vertex = _quantize_vertex(
                (float(match.group(1)), float(match.group(2)), float(match.group(3)))
            )
            vertices.append(vertex)
            current.append(vertex)
            if len(current) == 3:
                triangles.append((current[0], current[1], current[2]))
                current = []

    if not vertices:
        return {
            "vertex_count": 0,
            "triangle_count": 0,
            "unique_edge_count": 0,
            "bbox_min": [0.0, 0.0, 0.0],
            "bbox_max": [0.0, 0.0, 0.0],
            "bbox_size": [0.0, 0.0, 0.0],
        }

    unique_edges: set[tuple[tuple[float, float, float], tuple[float, float, float]]] = (
        set()
    )
    for a, b, c in triangles:
        for p, q in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((p, q)))
            unique_edges.add(cast(Any, edge))

    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    bbox_min = (min(xs), min(ys), min(zs))
    bbox_max = (max(xs), max(ys), max(zs))
    bbox_size = (
        round(bbox_max[0] - bbox_min[0], 6),
        round(bbox_max[1] - bbox_min[1], 6),
        round(bbox_max[2] - bbox_min[2], 6),
    )

    return {
        "vertex_count": len(set(vertices)),
        "triangle_count": len(triangles),
        "unique_edge_count": len(unique_edges),
        "bbox_min": list(bbox_min),
        "bbox_max": list(bbox_max),
        "bbox_size": list(bbox_size),
    }


@tool(args_schema=RenderInput)
def render_openscad(scad_code: str, image_sizes: list[str] | None = None) -> str:
    """Render OpenSCAD source into one or more annotated PNG images."""
    sizes = image_sizes or ["1024x768"]
    if not sizes:
        raise RuntimeError("image_sizes cannot be empty")

    RENDERS_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".scad", delete=False, encoding="utf-8"
    ) as temp_scad:
        temp_scad.write(scad_code)
        scad_path = Path(temp_scad.name)

    stl_path = scad_path.with_suffix(".stl")

    try:
        stl_cmd = [
            "openscad",
            "-o",
            str(stl_path),
            "--export-format",
            "asciistl",
            str(scad_path),
            "--render",
        ]
        subprocess.run(stl_cmd, check=True, text=True, capture_output=True)
        mesh_stats = _mesh_stats_from_ascii_stl(stl_path)

        renders: list[dict[str, str]] = []
        has_explicit_camera = "$vpt" in scad_code or "$vpd" in scad_code
        for size in sizes:
            width, height = _validate_size(size)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            safe_size = f"{width}x{height}"
            output_png = RENDERS_DIR / f"render_{safe_size}_{timestamp}.png"
            cmd = [
                "openscad",
                "-o",
                str(output_png),
                str(scad_path),
                "--render",
                f"--imgsize={width},{height}",
                "--projection=o",
                "--view=axes,scales,edges",
            ]
            if not has_explicit_camera:
                cmd.extend(["--autocenter", "--viewall"])
            proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
            log_text = proc.stdout.strip() or proc.stderr.strip() or "render complete"
            renders.append(
                {
                    "image_path": str(output_png),
                    "image_size": safe_size,
                    "mime_type": "image/png",
                    "openscad_log": log_text,
                }
            )

        payload = {
            "renders": renders,
            "annotation_mode": "deterministic_cli_overlays",
            "overlays": ["axes", "scales", "edges"],
            "mesh_stats": mesh_stats,
            "edge_counter": mesh_stats.get("unique_edge_count", 0),
        }
        return json.dumps(payload)
    except FileNotFoundError:
        raise RuntimeError(
            "openscad binary not found. Install OpenSCAD CLI and ensure 'openscad' is in PATH."
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "unknown OpenSCAD error").strip()
        raise RuntimeError(f"OpenSCAD render failed: {detail}")
    finally:
        scad_path.unlink(missing_ok=True)
        stl_path.unlink(missing_ok=True)


SYSTEM_PROMPT = (
    "You are an OpenSCAD design agent. You have exactly one tool: render_openscad. "
    "Always write valid OpenSCAD code and call render_openscad when rendering is needed. "
    "If the user asks for multiple shots/sizes, pass multiple values in image_sizes. "
    "Do not generate textual annotation geometry. The tool already applies deterministic "
    "overlays (axes/scales/edges) and edge counting."
)


def build_model() -> tuple[ChatOpenAI, Any]:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "")
    model_name = os.getenv("OPENAI_MODEL", "gpt-5")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env.")

    llm = ChatOpenAI(model=model_name, temperature=1)
    llm_with_tools = llm.bind_tools([render_openscad])
    return llm, llm_with_tools


def _tool_args(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        parsed = json.loads(raw_args)
        if isinstance(parsed, dict):
            return parsed
    raise RuntimeError("Invalid tool arguments returned by model.")


def run_agent_turn(llm: ChatOpenAI, llm_with_tools: Any, user_input: str) -> str:
    messages: list[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input),
    ]

    first = llm_with_tools.invoke(messages)
    messages.append(first)

    tool_calls = getattr(first, "tool_calls", None) or []
    if not tool_calls:
        if isinstance(first.content, str):
            return first.content
        return str(first.content)

    for call in tool_calls:
        if call.get("name") != "render_openscad":
            raise RuntimeError(f"Unexpected tool requested: {call.get('name')}")
        args = _tool_args(call.get("args"))
        tool_output = cast(Any, render_openscad).invoke(args)
        messages.append(ToolMessage(content=tool_output, tool_call_id=call["id"]))

        payload = json.loads(tool_output)
        renders = payload.get("renders")
        if not isinstance(renders, list) or not renders:
            raise RuntimeError("render_openscad output missing renders")

        content_parts: list[dict[str, Any]] = []
        edge_counter = payload.get("edge_counter", "unknown")
        mesh_stats = payload.get("mesh_stats", {})
        content_parts.append(
            {
                "type": "text",
                "text": (
                    "Rendered images attached. Analyze them visually. "
                    f"Deterministic overlays are enabled (axes/scales/edges). "
                    f"Universal edge count: {edge_counter}. Mesh stats: {mesh_stats}."
                ),
            }
        )

        for idx, render in enumerate(renders, start=1):
            if not isinstance(render, dict):
                continue
            image_path = Path(str(render.get("image_path", "")))
            image_size = str(render.get("image_size", "unknown"))
            if not image_path.exists():
                raise RuntimeError(f"Rendered image missing: {image_path}")
            image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
            data_uri = f"data:image/png;base64,{image_b64}"
            content_parts.append(
                {
                    "type": "text",
                    "text": f"Image {idx}: size={image_size}, path={image_path}",
                }
            )
            content_parts.append({"type": "image_url", "image_url": {"url": data_uri}})

        messages.append(HumanMessage(content=cast(Any, content_parts)))

    final = llm.invoke(messages)
    if isinstance(final.content, str):
        return final.content
    return str(final.content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LangChain OpenSCAD + GPT-5 vision agent"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Task for the agent. If omitted, starts interactive mode.",
    )
    args = parser.parse_args()

    llm, llm_with_tools = build_model()

    if args.prompt:
        print(run_agent_turn(llm, llm_with_tools, args.prompt))
        return

    print("OpenSCAD agent ready. Type 'exit' to quit.")
    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        print(run_agent_turn(llm, llm_with_tools, user_input))


if __name__ == "__main__":
    main()

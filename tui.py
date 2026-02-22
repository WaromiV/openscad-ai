import base64
import json
import os
import platform
import re
import shlex
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.widgets import Button, Footer, Header, RichLog, Static, TextArea

from agent import RENDERS_DIR, render_openscad


ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "reference_images"
SESSION_LOG_DIR = ROOT / "session_logs"
IMAGE_TAG_PREFIX = "[Image "
FIXED_RENDER_SIZE = "1600x1200"


@dataclass(frozen=True)
class ShotSpec:
    key: str
    label: str
    vpr: tuple[int, int, int]


BASE_SHOTS: tuple[ShotSpec, ...] = (
    ShotSpec("top_ne", "iso with top + NORTH EAST seen", (55, 0, 45)),
    ShotSpec("top_sw", "iso with top+SOUTH WEST", (55, 0, 225)),
    ShotSpec("bottom_ne", "iso with BOTTOM NORTH EAST", (-55, 0, 45)),
    ShotSpec("bottom_sw", "iso with BOTTOM SOUTH WEST", (-55, 0, 225)),
)
ZOOM_SHOT_LABEL = "zoomed x3 iso with top + NORTH EAST seen"
FIXED_SHOT_MANIFEST: tuple[str, ...] = (
    BASE_SHOTS[0].label,
    ZOOM_SHOT_LABEL,
    BASE_SHOTS[1].label,
    BASE_SHOTS[2].label,
    BASE_SHOTS[3].label,
)


DESIGN_SYSTEM_PROMPT = (
    "You are an expert OpenSCAD reverse-engineering assistant. "
    "Use the user prompt and attached reference images to produce OpenSCAD code that visually "
    "matches the target object as closely as possible. "
    "The render tool already adds deterministic overlays, so do not add annotation geometry in SCAD. "
    "When a syntax retry request is provided, prioritize fixing parser/runtime errors under the same iteration index. "
    "Use compact, valid OpenSCAD code."
)


REVIEW_SYSTEM_PROMPT = (
    "You are a strict visual reviewer. Compare generated renders against the attached reference images. "
    "Use highest effort to match target geometry. Stop only when match is strong. "
    "Ignore color discrepancies entirely. Render colors are diagnostic only and come from overlay/annotation display "
    "(edges, axes, scales), not material or texture intent. "
    "Prioritize geometry, proportions, silhouettes, hole placements, dimensions, and structural features. "
    "If mismatch remains, produce improved SCAD for next iteration. "
    "Keep reasoning_summary concise and do not output hidden chain-of-thought."
)


class DesignDraft(BaseModel):
    reasoning_summary: str = Field(..., description="Concise rationale summary.")
    scad_code: str = Field(..., description="OpenSCAD code candidate.")


class ReviewDraft(BaseModel):
    reasoning_summary: str = Field(..., description="Concise comparison summary.")
    decision: Literal["iterate", "stop"]
    match_score: int = Field(..., ge=0, le=100)
    issues: list[str] = Field(default_factory=list)
    next_scad_code: str | None = None


@dataclass
class IterationMemory:
    iteration: int
    scad_code: str
    shot_labels: list[str]
    render_payload: dict[str, Any]
    review: ReviewDraft | None = None


@dataclass
class Attachment:
    image_id: int
    path: Path

    @property
    def tag(self) -> str:
        return f"[Image {self.image_id}]"


class TagAwareTextArea(TextArea):
    TAG_AT_CURSOR = re.compile(r"(?:\s)?\[Image\s+\d+\]$")

    def action_delete_left(self) -> None:
        if not self.selection.is_empty:
            super().action_delete_left()
            return

        row, col = self.cursor_location
        if col <= 0:
            super().action_delete_left()
            return

        line_text = self.get_line(row).plain
        prefix = line_text[:col]
        match = self.TAG_AT_CURSOR.search(prefix)
        if match is None:
            super().action_delete_left()
            return

        self.delete((row, match.start()), (row, col), maintain_selection_offset=False)


def _image_data_uri(path: Path) -> str:
    image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{image_b64}"


def _append_reference_images(
    content: list[dict[str, Any]],
    refs: list[Attachment],
) -> None:
    for ref in refs:
        content.append(
            {
                "type": "text",
                "text": f"Reference {ref.tag} path={ref.path}",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_uri(ref.path)},
            }
        )


def _append_generated_history_images(
    content: list[dict[str, Any]],
    history: list[IterationMemory],
) -> None:
    for record in history:
        renders = record.render_payload.get("renders", [])
        for idx, render in enumerate(renders, start=1):
            if not isinstance(render, dict):
                continue
            render_path = Path(str(render.get("image_path", "")))
            if not render_path.exists():
                continue
            render_size = str(render.get("image_size", "unknown"))
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Historical Render iter={record.iteration} image={idx} "
                        f"size={render_size} path={render_path}"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_uri(render_path)},
                }
            )


def _history_summary_json(history: list[IterationMemory]) -> str:
    if not history:
        return "[]"

    summary: list[dict[str, Any]] = []
    for record in history:
        renders = record.render_payload.get("renders", [])
        render_summary: list[dict[str, Any]] = []
        for render in renders:
            if not isinstance(render, dict):
                continue
            render_summary.append(
                {
                    "image_path": str(render.get("image_path", "")),
                    "image_size": str(render.get("image_size", "")),
                }
            )

        review_obj: dict[str, Any] | None = None
        if record.review is not None:
            review_obj = {
                "decision": record.review.decision,
                "match_score": record.review.match_score,
                "issues": record.review.issues,
                "reasoning_summary": record.review.reasoning_summary,
            }

        summary.append(
            {
                "iteration": record.iteration,
                "shot_labels": record.shot_labels,
                "edge_counter": record.render_payload.get("edge_counter"),
                "mesh_stats": record.render_payload.get("mesh_stats"),
                "renders": render_summary,
                "scad_code": record.scad_code,
                "review": review_obj,
            }
        )

    return json.dumps(summary, indent=2)


def _to_multimodal_content(
    prompt_text: str, attachments: list[Attachment]
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    _append_reference_images(content, attachments)
    return content


def _to_review_content(
    user_prompt: str,
    refs: list[Attachment],
    iteration: int,
    scad_code: str,
    render_payload: dict[str, Any],
    prior_issues: list[str],
    history: list[IterationMemory],
) -> list[dict[str, Any]]:
    renders = render_payload.get("renders", [])
    edge_counter = render_payload.get("edge_counter", "unknown")
    mesh_stats = render_payload.get("mesh_stats", {})
    issues_text = "\n".join(f"- {issue}" for issue in prior_issues) or "- none"
    history_json = _history_summary_json(history)

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Review this iteration and decide whether to iterate or stop.\n"
                f"Iteration: {iteration}\n"
                f"User request: {user_prompt}\n"
                f"Prior issues:\n{issues_text}\n"
                f"Full prior iteration memory (code + images + ratings):\n{history_json}\n"
                "Important: ignore color differences. Colors are only diagnostic overlays for edges/annotations.\n"
                "Use baked top labels on each image as authoritative view identity.\n"
                f"Current edge_counter: {edge_counter}\n"
                f"Current mesh_stats: {mesh_stats}\n"
                "Current OpenSCAD code:\n"
                f"{scad_code}"
            ),
        }
    ]

    _append_reference_images(content, refs)
    _append_generated_history_images(content, history)

    for idx, render in enumerate(renders, start=1):
        if not isinstance(render, dict):
            continue
        render_path = Path(str(render.get("image_path", "")))
        if not render_path.exists():
            continue
        render_size = str(render.get("image_size", "unknown"))
        content.append(
            {
                "type": "text",
                "text": f"Generated Render {idx}: size={render_size}, path={render_path}",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_uri(render_path)},
            }
        )

    return content


def _to_designer_refinement_content(
    user_prompt: str,
    refs: list[Attachment],
    history: list[IterationMemory],
    last_review: ReviewDraft,
) -> list[dict[str, Any]]:
    history_json = _history_summary_json(history)
    issues_text = "\n".join(f"- {issue}" for issue in last_review.issues) or "- none"
    reviewer_code_hint = last_review.next_scad_code or "(none)"

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Generate an improved OpenSCAD candidate for the next iteration.\n"
                f"User request: {user_prompt}\n"
                f"Last reviewer decision: {last_review.decision}\n"
                f"Last reviewer match_score: {last_review.match_score}\n"
                f"Last reviewer reasoning_summary: {last_review.reasoning_summary}\n"
                f"Last reviewer issues:\n{issues_text}\n"
                f"Reviewer suggested next_scad_code (optional hint):\n{reviewer_code_hint}\n"
                "Use the full memory below (all prior code, renders, and ratings).\n"
                f"{history_json}\n"
                "Return improved scad_code only. Shot generation is deterministic and fixed by runtime logic."
            ),
        }
    ]

    _append_reference_images(content, refs)
    _append_generated_history_images(content, history)
    return content


def _to_syntax_retry_content(
    user_prompt: str,
    refs: list[Attachment],
    history: list[IterationMemory],
    iteration: int,
    syntax_retry_index: int,
    last_scad_code: str,
    fixed_shots: list[str],
    error_text: str,
) -> list[dict[str, Any]]:
    history_json = _history_summary_json(history)
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Syntax retry required for the same iteration index.\n"
                f"User request: {user_prompt}\n"
                f"Iteration: {iteration}\n"
                f"Syntax retry: {syntax_retry_index}\n"
                f"Fixed shot manifest: {fixed_shots}\n"
                "The previous SCAD failed to parse/render.\n"
                f"OpenSCAD error:\n{error_text}\n"
                "Previous SCAD:\n"
                f"{last_scad_code}\n"
                "Return corrected syntactically valid OpenSCAD code. "
                "Do not change intent, only fix and improve as needed.\n"
                "Full prior memory (code + renders + ratings):\n"
                f"{history_json}"
            ),
        }
    ]

    _append_reference_images(content, refs)
    _append_generated_history_images(content, history)
    return content


def _view_scad(scad_code: str, vpr: tuple[int, int, int]) -> str:
    rx, ry, rz = vpr
    return f"$vpr=[{rx},{ry},{rz}];\n{scad_code}\n"


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
    return output


def _render_fixed_shots(scad_code: str) -> dict[str, Any]:
    shot_outputs: dict[str, dict[str, Any]] = {}
    first_payload: dict[str, Any] | None = None

    for shot in BASE_SHOTS:
        oriented_scad = _view_scad(scad_code, shot.vpr)
        try:
            tool_output = cast(Any, render_openscad).invoke(
                {"scad_code": oriented_scad, "image_sizes": [FIXED_RENDER_SIZE]}
            )
        except Exception as exc:
            raise RuntimeError(f"Shot '{shot.label}' failed: {exc}")

        payload = json.loads(tool_output)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Shot '{shot.label}' returned non-dict payload")
        renders = payload.get("renders")
        if not isinstance(renders, list) or not renders:
            raise RuntimeError(f"Shot '{shot.label}' returned no renders")

        render = renders[0]
        if not isinstance(render, dict):
            raise RuntimeError(f"Shot '{shot.label}' render entry malformed")

        image_path = Path(str(render.get("image_path", "")))
        if not image_path.exists():
            raise RuntimeError(f"Shot '{shot.label}' image missing: {image_path}")

        _bake_top_label(image_path, shot.label)
        shot_outputs[shot.key] = {
            "image_path": str(image_path),
            "image_size": str(render.get("image_size", FIXED_RENDER_SIZE)),
            "mime_type": str(render.get("mime_type", "image/png")),
            "openscad_log": str(render.get("openscad_log", "")),
            "view_key": shot.key,
            "view_label": shot.label,
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
        "image_size": str(top_ne.get("image_size", FIXED_RENDER_SIZE)),
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
    ]

    first_payload = first_payload or {}
    return {
        "renders": ordered_renders,
        "annotation_mode": "deterministic_cli_overlays_with_fixed_view_labels",
        "overlays": ["axes", "scales", "edges", "top_label"],
        "edge_counter": first_payload.get("edge_counter"),
        "mesh_stats": first_payload.get("mesh_stats"),
        "shot_policy": "fixed_5_shot_manifest_v1",
        "shot_manifest": [render["view_label"] for render in ordered_renders],
    }


class OpenSCADRebuildTUI(App[None]):
    BINDINGS = [
        Binding("ctrl+enter", "send_prompt", "Send"),
        Binding("ctrl+v", "paste_image", "Paste Image", priority=True),
        Binding(
            "ctrl+shift+v", "paste_image", "Paste Image", show=False, priority=True
        ),
        Binding(
            "shift+insert", "paste_image", "Paste Image", show=False, priority=True
        ),
        Binding("ctrl+c", "interrupt", "Stop / Exit", priority=True),
        Binding("ctrl+backspace", "remove_last_image", "Remove Last Image"),
    ]

    CSS = """
    Screen {
        layout: vertical;
    }

    #main {
        height: 1fr;
    }

    #chat_log {
        width: 3fr;
        border: round #5fb3ff;
    }

    #sidebar {
        width: 2fr;
        border: round #6bd18f;
    }

    #attachments {
        height: 1fr;
        border: round #6bd18f;
        padding: 1 1;
    }

    #status {
        height: 5;
        border: round #ffcc66;
        padding: 1 1;
    }

    #composer {
        height: 14;
        border: round #9e8bff;
        padding: 1 1;
    }

    #prompt_box {
        height: 1fr;
    }

    #controls {
        height: 3;
    }

    Button {
        margin-right: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.attachments: dict[int, Attachment] = {}
        self.next_image_id = 1
        self.max_iterations = 8
        self.max_syntax_retries = 8
        self.run_in_progress = False
        self.stop_requested = False
        self.exit_armed = False
        self.last_escape_at = 0.0
        self.escape_window_seconds = 1.0
        self.escape_press_count = 0
        self.reconstruction_worker: Any | None = None
        self.suppress_prompt_tag_sync = False
        self.last_prompt_tag_ids: set[int] = set()
        self.session_log_path: Path | None = None
        self.session_log_data: dict[str, Any] = {}
        self.session_event_seq = 0
        self.session_log_lock = threading.Lock()
        self.run_counter = 0
        self.llm: ChatOpenAI | None = None
        self.designer: Any = None
        self.reviewer: Any = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            yield RichLog(id="chat_log", markup=False, wrap=True, auto_scroll=True)
            with Vertical(id="sidebar"):
                yield Static("No attachments yet.", id="attachments")
                yield Static("Idle", id="status")
        with Vertical(id="composer"):
            yield TagAwareTextArea(
                "",
                id="prompt_box",
            )
            with Horizontal(id="controls"):
                yield Button("Send", id="send", variant="success")
                yield Button("Paste Clipboard Image", id="paste")
                yield Button("Remove Last Image", id="remove")
                yield Button("Clear Prompt", id="clear")
        yield Footer()

    def on_mount(self) -> None:
        load_dotenv()
        model_name = os.getenv("OPENAI_MODEL", "gpt-5-nano")
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env.")

        self._init_session_log(model_name)

        self.llm = ChatOpenAI(model=model_name, temperature=1)
        self.designer = self.llm.with_structured_output(DesignDraft)
        self.reviewer = self.llm.with_structured_output(ReviewDraft)

        self._log_panel(
            "System",
            (
                "Rich OpenSCAD TUI ready.\n"
                "- Paste image with Ctrl+V or the button.\n"
                "- Images are tagged like [Image 1] and inserted into prompt.\n"
                "- You can type tags manually, remove tags, and iterate automatically.\n"
                "- Esc Esc or Ctrl+C: first stops run, second exits TUI.\n"
                "- Inline commands: /attach <path ...>, /remove <id>.\n"
                f"- Full JSON session logs: {self.session_log_path}"
            ),
            "cyan",
        )
        self._set_status("Idle")
        self._refresh_attachments_view()

    def on_unmount(self) -> None:
        self._append_session_event(
            "session_closed",
            {
                "message": "TUI closed",
            },
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "send":
            self.action_send_prompt()
        elif button_id == "paste":
            self.action_paste_image()
        elif button_id == "remove":
            self.action_remove_last_image()
        elif button_id == "clear":
            self._set_prompt_text("", preserve_attachments=False)

    def on_key(self, event: Key) -> None:
        if event.key in {"ctrl+v", "ctrl+shift+v", "shift+insert"}:
            self.action_paste_image()
            event.stop()
            event.prevent_default()
            return

        if event.key == "escape":
            self._handle_escape_key()
            event.stop()
            event.prevent_default()
            return

    def action_interrupt(self) -> None:
        self._handle_interrupt("Ctrl+C")

    def _handle_escape_key(self) -> None:
        now = time.monotonic()
        if now - self.last_escape_at <= self.escape_window_seconds:
            self.escape_press_count += 1
        else:
            self.escape_press_count = 1
        self.last_escape_at = now

        if self.escape_press_count >= 2:
            self.escape_press_count = 0
            self._handle_interrupt("Esc Esc")
            return

        if self.run_in_progress:
            self._set_status("Press Esc again quickly to stop current execution.")
        else:
            self._set_status("Press Esc again quickly to arm stop/exit.")

    def _handle_interrupt(self, trigger: str) -> None:
        self._append_session_event(
            "interrupt",
            {
                "trigger": trigger,
                "run_in_progress": self.run_in_progress,
                "stop_requested": self.stop_requested,
                "exit_armed": self.exit_armed,
            },
        )
        if self.run_in_progress:
            if not self.stop_requested:
                self.stop_requested = True
                worker = self.reconstruction_worker
                if worker is not None:
                    try:
                        worker.cancel()
                    except Exception:
                        pass
                self.exit_armed = True
                self._log_panel(
                    "Interrupt",
                    f"{trigger}: stop requested. Press Esc Esc or Ctrl+C again to exit TUI.",
                    "yellow",
                )
                self._set_status(
                    "Stopping execution... Press Esc Esc/Ctrl+C again to exit."
                )
            else:
                self._log_panel(
                    "Interrupt",
                    f"{trigger}: second interrupt received. Exiting TUI now.",
                    "red",
                )
                self.exit()
            return

        if self.exit_armed:
            self._log_panel("Exit", f"{trigger}: exiting TUI.", "yellow")
            self.exit()
            return

        self.exit_armed = True
        self._log_panel(
            "Interrupt",
            f"{trigger}: no active execution. Press Esc Esc or Ctrl+C again to exit.",
            "yellow",
        )
        self._set_status("Exit armed. Press Esc Esc/Ctrl+C again to exit.")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "prompt_box":
            return

        current_ids = self._extract_tag_ids(event.text_area.text)
        if self.suppress_prompt_tag_sync:
            self.last_prompt_tag_ids = current_ids
            return

        removed_ids = sorted(self.last_prompt_tag_ids - current_ids)
        for image_id in removed_ids:
            self._detach_attachment_by_id(image_id, reason="Tag removed from prompt")

        self.last_prompt_tag_ids = current_ids

    def action_send_prompt(self) -> None:
        if self.run_in_progress:
            self._set_status("A run is already in progress.")
            return

        prompt_box = self.query_one("#prompt_box", TextArea)
        raw_prompt = prompt_box.text.strip()
        if not raw_prompt:
            self._set_status("Prompt is empty.")
            return

        prompt = self._process_inline_commands(raw_prompt)
        if not prompt:
            self._set_status("Only commands were processed. Add a prompt.")
            prompt_box.text = ""
            return

        refs = self._resolve_tagged_attachments(prompt)
        self.run_counter += 1
        run_id = self.run_counter
        self._append_session_event(
            "run_requested",
            {
                "run_id": run_id,
                "prompt": prompt,
                "referenced_image_ids": [ref.image_id for ref in refs],
                "referenced_paths": [str(ref.path) for ref in refs],
            },
        )
        self._log_panel("User Prompt", prompt, "green")
        if refs:
            tags = ", ".join(ref.tag for ref in refs)
            self._log_panel("Referenced Images", tags, "green")
        else:
            self._log_panel(
                "Referenced Images",
                "No image tags found in prompt. Running text-only design loop.",
                "yellow",
            )

        self._set_prompt_text("", preserve_attachments=True)
        self.exit_armed = False
        self.stop_requested = False
        self.escape_press_count = 0
        self.last_escape_at = 0.0
        self.run_in_progress = True
        self._set_status("Running reconstruction loop...")
        self.reconstruction_worker = self.run_reconstruction(prompt, refs, run_id)

    def action_paste_image(self) -> None:
        try:
            from PIL import Image, ImageGrab
        except Exception:
            self._log_panel(
                "Clipboard",
                "Pillow clipboard support is unavailable. Install pillow or use /attach <path>.",
                "red",
            )
            return

        clip = ImageGrab.grabclipboard()
        if clip is None:
            self._set_status("Clipboard does not contain an image.")
            self._append_session_event("clipboard_empty")
            return

        if isinstance(clip, Image.Image):
            REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
            filename = f"clipboard_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.png"
            image_path = REFERENCE_DIR / filename
            clip.save(image_path, format="PNG")
            self._attach_image_path(image_path, insert_tag=True)
            self._append_session_event(
                "clipboard_image_saved",
                {"path": str(image_path)},
            )
            return

        if isinstance(clip, list):
            attached = 0
            for raw in cast(list[Any], clip):
                path = Path(str(raw)).expanduser().resolve()
                if path.is_file() and path.suffix.lower() in {
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp",
                    ".bmp",
                }:
                    self._attach_image_path(path, insert_tag=True)
                    attached += 1
            if attached:
                self._set_status(f"Attached {attached} image(s) from clipboard list.")
                self._append_session_event(
                    "clipboard_file_list_attached",
                    {"count": attached},
                )
            else:
                self._set_status(
                    "Clipboard file list did not contain supported images."
                )
                self._append_session_event("clipboard_file_list_no_images")
            return

        self._set_status("Clipboard format unsupported for image paste.")
        self._append_session_event("clipboard_unsupported", {"type": str(type(clip))})

    def action_remove_last_image(self) -> None:
        if not self.attachments:
            self._set_status("No attachments to remove.")
            return
        last_id = max(self.attachments.keys())
        self._detach_attachment_by_id(last_id, reason="Removed from control button")

    @work(thread=True)
    def run_reconstruction(
        self,
        user_prompt: str,
        refs: list[Attachment],
        run_id: int,
    ) -> None:
        try:
            if self.llm is None or self.designer is None or self.reviewer is None:
                raise RuntimeError("Model is not initialized.")

            self._append_session_event(
                "run_started",
                {
                    "run_id": run_id,
                    "prompt": user_prompt,
                    "refs": [
                        {"image_id": ref.image_id, "path": str(ref.path)}
                        for ref in refs
                    ],
                    "max_iterations": self.max_iterations,
                    "max_syntax_retries": self.max_syntax_retries,
                    "memory_mode": "persistent_multimodal_both_agents",
                    "fixed_shot_manifest": list(FIXED_SHOT_MANIFEST),
                    "fixed_render_size": FIXED_RENDER_SIZE,
                },
            )

            def should_stop() -> bool:
                worker = self.reconstruction_worker
                if self.stop_requested:
                    return True
                if worker is not None and bool(getattr(worker, "is_cancelled", False)):
                    return True
                return False

            if should_stop():
                self.call_from_thread(
                    self._log_panel,
                    "Loop Stopped",
                    "Execution stopped before first iteration.",
                    "yellow",
                )
                return

            initial_content = _to_multimodal_content(user_prompt, refs)
            designer_messages: list[Any] = [SystemMessage(content=DESIGN_SYSTEM_PROMPT)]
            reviewer_messages: list[Any] = [SystemMessage(content=REVIEW_SYSTEM_PROMPT)]
            history: list[IterationMemory] = []

            designer_messages.append(HumanMessage(content=cast(Any, initial_content)))
            initial = self.designer.invoke(designer_messages)
            designer_messages.append(
                AIMessage(content=json.dumps(initial.model_dump(), indent=2))
            )
            self._append_session_event(
                "initial_design",
                {
                    "run_id": run_id,
                    "reasoning_summary": initial.reasoning_summary,
                    "scad_code": initial.scad_code,
                },
            )
            if should_stop():
                self.call_from_thread(
                    self._log_panel,
                    "Loop Stopped",
                    "Execution stopped after initial design step.",
                    "yellow",
                )
                return
            scad_code = initial.scad_code

            self.call_from_thread(
                self._log_iteration_proposal,
                1,
                initial.reasoning_summary,
                list(FIXED_SHOT_MANIFEST),
                scad_code,
            )

            prior_issues: list[str] = []
            final_payload: dict[str, Any] | None = None
            stop_iteration: int | None = None

            for iteration in range(1, self.max_iterations + 1):
                if should_stop():
                    self.call_from_thread(
                        self._log_panel,
                        "Loop Stopped",
                        f"Execution stopped by user at iteration {iteration}.",
                        "yellow",
                    )
                    return

                syntax_retry_index = 0
                payload: dict[str, Any] | None = None

                while True:
                    if should_stop():
                        self.call_from_thread(
                            self._log_panel,
                            "Loop Stopped",
                            (
                                f"Execution stopped by user during syntax retry loop "
                                f"at iteration {iteration}."
                            ),
                            "yellow",
                        )
                        return

                    args = {
                        "scad_code": "<omitted for brevity>",
                        "fixed_shot_manifest": list(FIXED_SHOT_MANIFEST),
                        "fixed_render_size": FIXED_RENDER_SIZE,
                    }
                    panel_title = f"Iteration {iteration} - Tool Call"
                    event_kind = "iteration_tool_call"
                    if syntax_retry_index > 0:
                        panel_title = f"Iteration {iteration} - Syntax Retry {syntax_retry_index} Tool Call"
                        event_kind = "iteration_syntax_retry_tool_call"

                    self._append_session_event(
                        event_kind,
                        {
                            "run_id": run_id,
                            "iteration": iteration,
                            "syntax_retry_index": syntax_retry_index,
                            "args": args,
                        },
                    )
                    self.call_from_thread(
                        self._log_panel,
                        panel_title,
                        json.dumps(args, indent=2),
                        "blue",
                    )

                    try:
                        payload = _render_fixed_shots(scad_code)
                        final_payload = payload
                        self._append_session_event(
                            "iteration_tool_result",
                            {
                                "run_id": run_id,
                                "iteration": iteration,
                                "syntax_retry_index": syntax_retry_index,
                                "payload": payload,
                            },
                        )
                        if syntax_retry_index > 0:
                            self.call_from_thread(
                                self._log_panel,
                                f"Iteration {iteration} - Syntax Retry Success",
                                (
                                    f"Render succeeded after {syntax_retry_index} syntax "
                                    "retry attempt(s)."
                                ),
                                "green",
                            )
                        self.call_from_thread(self._log_tool_result, iteration, payload)
                        break
                    except Exception as tool_exc:
                        error_text = f"{tool_exc}\n\n{traceback.format_exc()}"
                        self._append_session_event(
                            "iteration_syntax_retry_error",
                            {
                                "run_id": run_id,
                                "iteration": iteration,
                                "syntax_retry_index": syntax_retry_index + 1,
                                "error": str(tool_exc),
                            },
                        )
                        self.call_from_thread(
                            self._log_panel,
                            f"Iteration {iteration} - Syntax Retry",
                            (
                                f"Syntax retry {syntax_retry_index + 1} triggered.\n"
                                "Tool failure returned to generator for correction.\n\n"
                                f"{tool_exc}"
                            ),
                            "red",
                        )

                        syntax_retry_index += 1
                        if syntax_retry_index > self.max_syntax_retries:
                            raise RuntimeError(
                                (
                                    "Exceeded syntax retry limit "
                                    f"({self.max_syntax_retries}) at iteration {iteration}. "
                                    "Last tool error:\n"
                                    f"{tool_exc}"
                                )
                            )

                        retry_content = _to_syntax_retry_content(
                            user_prompt=user_prompt,
                            refs=refs,
                            history=history,
                            iteration=iteration,
                            syntax_retry_index=syntax_retry_index,
                            last_scad_code=scad_code,
                            fixed_shots=list(FIXED_SHOT_MANIFEST),
                            error_text=error_text,
                        )
                        designer_messages.append(
                            HumanMessage(content=cast(Any, retry_content))
                        )
                        retry_design = self.designer.invoke(designer_messages)
                        designer_messages.append(
                            AIMessage(
                                content=json.dumps(retry_design.model_dump(), indent=2)
                            )
                        )
                        self._append_session_event(
                            "iteration_syntax_retry_generation",
                            {
                                "run_id": run_id,
                                "iteration": iteration,
                                "syntax_retry_index": syntax_retry_index,
                                "reasoning_summary": retry_design.reasoning_summary,
                                "scad_code": retry_design.scad_code,
                            },
                        )

                        retry_scad_code = retry_design.scad_code.strip()
                        if not retry_scad_code:
                            raise RuntimeError(
                                (
                                    "Syntax retry failed because generator returned empty "
                                    "scad_code."
                                )
                            )

                        scad_code = retry_scad_code
                        self.call_from_thread(
                            self._log_iteration_proposal,
                            iteration,
                            (
                                f"syntax retry #{syntax_retry_index}: "
                                f"{retry_design.reasoning_summary}"
                            ),
                            list(FIXED_SHOT_MANIFEST),
                            scad_code,
                        )

                if payload is None:
                    raise RuntimeError(
                        f"Iteration {iteration} did not produce a valid payload after syntax retries."
                    )

                if should_stop():
                    self.call_from_thread(
                        self._log_panel,
                        "Loop Stopped",
                        f"Execution stopped by user after tool call in iteration {iteration}.",
                        "yellow",
                    )
                    return

                review_content = _to_review_content(
                    user_prompt=user_prompt,
                    refs=refs,
                    iteration=iteration,
                    scad_code=scad_code,
                    render_payload=payload,
                    prior_issues=prior_issues,
                    history=history,
                )
                reviewer_messages.append(
                    HumanMessage(content=cast(Any, review_content))
                )
                review = self.reviewer.invoke(reviewer_messages)
                reviewer_messages.append(
                    AIMessage(content=json.dumps(review.model_dump(), indent=2))
                )
                self._append_session_event(
                    "iteration_review",
                    {
                        "run_id": run_id,
                        "iteration": iteration,
                        "review": review,
                    },
                )
                self.call_from_thread(self._log_review_result, iteration, review)

                record = IterationMemory(
                    iteration=iteration,
                    scad_code=scad_code,
                    shot_labels=list(FIXED_SHOT_MANIFEST),
                    render_payload=payload,
                    review=review,
                )
                history.append(record)
                self._append_session_event(
                    "iteration_memory_updated",
                    {
                        "run_id": run_id,
                        "iteration": iteration,
                        "history_size": len(history),
                    },
                )

                if should_stop():
                    self.call_from_thread(
                        self._log_panel,
                        "Loop Stopped",
                        f"Execution stopped by user after review in iteration {iteration}.",
                        "yellow",
                    )
                    return

                if review.decision == "stop":
                    stop_iteration = iteration
                    break

                if should_stop():
                    self.call_from_thread(
                        self._log_panel,
                        "Loop Stopped",
                        f"Execution stopped before design refinement for iteration {iteration + 1}.",
                        "yellow",
                    )
                    return

                refine_content = _to_designer_refinement_content(
                    user_prompt=user_prompt,
                    refs=refs,
                    history=history,
                    last_review=review,
                )
                designer_messages.append(
                    HumanMessage(content=cast(Any, refine_content))
                )
                next_design = self.designer.invoke(designer_messages)
                designer_messages.append(
                    AIMessage(content=json.dumps(next_design.model_dump(), indent=2))
                )
                self._append_session_event(
                    "iteration_next_design",
                    {
                        "run_id": run_id,
                        "iteration": iteration + 1,
                        "reasoning_summary": next_design.reasoning_summary,
                        "scad_code": next_design.scad_code,
                    },
                )

                next_scad_code = next_design.scad_code.strip()
                if not next_scad_code and review.next_scad_code:
                    next_scad_code = review.next_scad_code.strip()

                if not next_scad_code:
                    self.call_from_thread(
                        self._log_panel,
                        "Iteration Halted",
                        "No next_scad_code available from reviewer/designer for next iteration.",
                        "red",
                    )
                    break

                scad_code = next_scad_code
                prior_issues = review.issues

                self.call_from_thread(
                    self._log_iteration_proposal,
                    iteration + 1,
                    next_design.reasoning_summary,
                    list(FIXED_SHOT_MANIFEST),
                    scad_code,
                )

            if stop_iteration is None:
                self.call_from_thread(
                    self._log_panel,
                    "Loop Ended",
                    f"Reached max iterations ({self.max_iterations}) or halted by guard.",
                    "yellow",
                )
            else:
                self.call_from_thread(
                    self._log_panel,
                    "Loop Ended",
                    f"Model stopped automatically at iteration {stop_iteration}.",
                    "green",
                )

            output_dir = ROOT / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            scad_file = (
                output_dir
                / f"final_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.scad"
            )
            scad_file.write_text(scad_code, encoding="utf-8")

            final_summary = {
                "final_scad": str(scad_file),
                "fixed_shot_manifest": list(FIXED_SHOT_MANIFEST),
                "fixed_render_size": FIXED_RENDER_SIZE,
                "final_render_paths": [
                    render.get("image_path")
                    for render in (final_payload or {}).get("renders", [])
                    if isinstance(render, dict)
                ],
                "edge_counter": (final_payload or {}).get("edge_counter"),
                "mesh_stats": (final_payload or {}).get("mesh_stats"),
                "iterations_recorded": len(history),
                "history": json.loads(_history_summary_json(history)),
            }
            self.call_from_thread(
                self._log_panel,
                "Final Output",
                json.dumps(final_summary, indent=2),
                "green",
            )
            self._append_session_event(
                "run_finished",
                {
                    "run_id": run_id,
                    "final_summary": final_summary,
                    "stopped_iteration": stop_iteration,
                },
            )
        except Exception as exc:
            self._append_session_event(
                "run_error",
                {
                    "run_id": run_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            self.call_from_thread(
                self._log_panel,
                "Error",
                f"{exc}\n\n{traceback.format_exc()}",
                "red",
            )
        finally:
            self.call_from_thread(self._set_status, "Idle")
            self.reconstruction_worker = None
            self.stop_requested = False
            self.run_in_progress = False

    def _log_iteration_proposal(
        self,
        iteration: int,
        summary: str,
        shot_manifest: list[str],
        scad_code: str,
    ) -> None:
        shot_lines = "\n".join(f"- {shot}" for shot in shot_manifest)
        text = (
            f"Reasoning summary:\n{summary}\n\n"
            "Deterministic fixed shots (model has no control):\n"
            f"{shot_lines}\n"
            f"Render size: {FIXED_RENDER_SIZE}\n"
            "(Reasoning is shown as concise summary, not raw chain-of-thought.)"
        )
        self._log_panel(f"Iteration {iteration} - Model Proposal", text, "magenta")
        self._log_syntax_panel(
            f"Iteration {iteration} - OpenSCAD", scad_code, "scad", "magenta"
        )

    def _log_tool_result(self, iteration: int, payload: dict[str, Any]) -> None:
        renders = payload.get("renders", [])
        paths: list[str] = []
        for render in renders:
            if isinstance(render, dict):
                paths.append(str(render.get("image_path", "")))

        body = {
            "annotation_mode": payload.get("annotation_mode"),
            "overlays": payload.get("overlays"),
            "edge_counter": payload.get("edge_counter"),
            "mesh_stats": payload.get("mesh_stats"),
            "render_paths": paths,
        }
        self._log_panel(
            f"Iteration {iteration} - Tool Result",
            json.dumps(body, indent=2),
            "blue",
        )

    def _log_review_result(self, iteration: int, review: ReviewDraft) -> None:
        issues_text = "\n".join(f"- {x}" for x in review.issues) or "- none"
        body = (
            f"Decision: {review.decision}\n"
            f"Match score: {review.match_score}\n"
            f"Reasoning summary:\n{review.reasoning_summary}\n\n"
            f"Issues:\n{issues_text}"
        )
        self._log_panel(f"Iteration {iteration} - Review", body, "yellow")

    def _utc_timestamp(self) -> str:
        return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): self._json_safe(inner) for key, inner in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(item) for item in value]
        if hasattr(value, "model_dump"):
            try:
                return self._json_safe(value.model_dump())
            except Exception:
                return str(value)
        return str(value)

    def _flush_session_log_unlocked(self) -> None:
        if self.session_log_path is None:
            return
        self.session_log_path.write_text(
            json.dumps(self.session_log_data, indent=2),
            encoding="utf-8",
        )

    def _append_session_event(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self.session_log_lock:
            self.session_event_seq += 1
            event: dict[str, Any] = {
                "seq": self.session_event_seq,
                "ts": self._utc_timestamp(),
                "kind": kind,
            }
            if payload:
                event["payload"] = self._json_safe(payload)

            self.session_log_data.setdefault("events", []).append(event)
            self.session_log_data["last_updated_at"] = self._utc_timestamp()
            self._flush_session_log_unlocked()

    def _init_session_log(self, model_name: str) -> None:
        SESSION_LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        self.session_log_path = SESSION_LOG_DIR / f"tui_session_{stamp}.json"
        self.session_log_data = {
            "schema": "openscad_rebuild_tui_session_log.v1",
            "session_id": stamp,
            "started_at": self._utc_timestamp(),
            "app": "OpenSCADRebuildTUI",
            "cwd": str(ROOT),
            "model": model_name,
            "python": sys.version,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "events": [],
        }
        with self.session_log_lock:
            self._flush_session_log_unlocked()

        self._append_session_event(
            "session_opened",
            {
                "message": "TUI opened",
                "session_log_path": str(self.session_log_path),
            },
        )

    def _log_panel(self, title: str, body: str, color: str) -> None:
        log = self.query_one("#chat_log", RichLog)
        panel = Panel(Text(body), title=title, border_style=color)
        log.write(panel)
        self._append_session_event(
            "ui_panel",
            {
                "title": title,
                "color": color,
                "body": body,
            },
        )

    def _log_syntax_panel(
        self,
        title: str,
        code: str,
        language: str,
        color: str,
    ) -> None:
        log = self.query_one("#chat_log", RichLog)
        syntax = Syntax(code, language, line_numbers=True, word_wrap=True)
        log.write(Panel(syntax, title=title, border_style=color))
        self._append_session_event(
            "ui_code_panel",
            {
                "title": title,
                "language": language,
                "color": color,
                "code": code,
            },
        )

    def _extract_tag_ids(self, text: str) -> set[int]:
        ids: set[int] = set()
        for match in re.finditer(r"\[Image\s+(\d+)\]", text):
            raw = match.group(1)
            if raw.isdigit():
                ids.add(int(raw))
        return ids

    def _set_prompt_text(self, text: str, preserve_attachments: bool) -> None:
        prompt_box = self.query_one("#prompt_box", TextArea)
        if preserve_attachments:
            self.suppress_prompt_tag_sync = True
        prompt_box.text = text
        self.last_prompt_tag_ids = self._extract_tag_ids(text)
        if preserve_attachments:
            self.suppress_prompt_tag_sync = False

    def _detach_attachment_by_id(self, image_id: int, reason: str) -> None:
        attachment = self.attachments.pop(image_id, None)
        if attachment is None:
            return

        self._append_session_event(
            "attachment_detached",
            {
                "image_id": attachment.image_id,
                "path": str(attachment.path),
                "reason": reason,
            },
        )

        prompt_box = self.query_one("#prompt_box", TextArea)
        if attachment.tag in prompt_box.text:
            self.suppress_prompt_tag_sync = True
            updated = prompt_box.text.replace(attachment.tag, "").replace("  ", " ")
            prompt_box.text = updated
            self.last_prompt_tag_ids = self._extract_tag_ids(updated)
            self.suppress_prompt_tag_sync = False

        self._refresh_attachments_view()
        self._log_panel(
            "Attachment Removed",
            f"Removed {attachment.tag}: {attachment.path}\nReason: {reason}",
            "yellow",
        )

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)
        self._append_session_event("status", {"text": text})

    def _refresh_attachments_view(self) -> None:
        if not self.attachments:
            self.query_one("#attachments", Static).update("No attachments yet.")
            return

        lines: list[str] = []
        for image_id in sorted(self.attachments):
            attachment = self.attachments[image_id]
            lines.append(f"{attachment.tag} -> {attachment.path}")
        self.query_one("#attachments", Static).update("\n".join(lines))

    def _process_inline_commands(self, raw_prompt: str) -> str:
        out_lines: list[str] = []
        for line in raw_prompt.splitlines():
            stripped = line.strip()
            if stripped.startswith("/attach "):
                args = shlex.split(stripped[len("/attach ") :])
                for arg in args:
                    self._attach_image_path(Path(arg).expanduser(), insert_tag=True)
                continue
            if stripped.startswith("/remove "):
                raw = stripped[len("/remove ") :].strip()
                if raw.isdigit():
                    image_id = int(raw)
                    self._detach_attachment_by_id(
                        image_id,
                        reason="Removed by /remove command",
                    )
                continue
            out_lines.append(line)
        return "\n".join(out_lines).strip()

    def _attach_image_path(self, path: Path, insert_tag: bool) -> None:
        real_path = path.resolve()
        if not real_path.exists() or not real_path.is_file():
            self._log_panel("Attach Failed", f"File not found: {real_path}", "red")
            self._append_session_event(
                "attachment_failed",
                {
                    "path": str(real_path),
                    "reason": "file_not_found",
                },
            )
            return

        if real_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            self._log_panel(
                "Attach Failed",
                f"Unsupported image format: {real_path.suffix}",
                "red",
            )
            self._append_session_event(
                "attachment_failed",
                {
                    "path": str(real_path),
                    "reason": "unsupported_format",
                    "suffix": real_path.suffix.lower(),
                },
            )
            return

        image_id = self.next_image_id
        self.next_image_id += 1
        attachment = Attachment(image_id=image_id, path=real_path)
        self.attachments[image_id] = attachment
        self._refresh_attachments_view()
        self._append_session_event(
            "attachment_added",
            {
                "image_id": attachment.image_id,
                "path": str(attachment.path),
                "insert_tag": insert_tag,
            },
        )

        if insert_tag:
            prompt_box = self.query_one("#prompt_box", TextArea)
            prompt_box.insert(f" {attachment.tag}")

        self._log_panel(
            "Attachment Added", f"{attachment.tag}: {attachment.path}", "green"
        )

    def _resolve_tagged_attachments(self, prompt: str) -> list[Attachment]:
        refs: list[Attachment] = []
        seen: set[int] = set()

        i = 0
        while i < len(prompt):
            if prompt.startswith(IMAGE_TAG_PREFIX, i):
                end = prompt.find("]", i + len(IMAGE_TAG_PREFIX))
                if end != -1:
                    raw_id = prompt[i + len(IMAGE_TAG_PREFIX) : end].strip()
                    if raw_id.isdigit():
                        image_id = int(raw_id)
                        if image_id in self.attachments and image_id not in seen:
                            refs.append(self.attachments[image_id])
                            seen.add(image_id)
                    i = end + 1
                    continue
            i += 1

        return refs


def main() -> None:
    app = OpenSCADRebuildTUI()
    app.run()


if __name__ == "__main__":
    main()

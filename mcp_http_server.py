import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from mcp_server import MCPError, _error_response, _handle_request


class MCPHttpHandler(BaseHTTPRequestHandler):
    server_version = "openscad-mcp-http/0.1.0"

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/mcp":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        request_id: Any = None
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                self.send_error(HTTPStatus.BAD_REQUEST, "Missing request body")
                return

            raw = self.rfile.read(content_length)
            request = json.loads(raw)
            if not isinstance(request, dict):
                raise MCPError(-32600, "Request body must be a JSON object")

            request_id = request.get("id")
            response = _handle_request(request)

            if response is None:
                self.send_response(HTTPStatus.ACCEPTED)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            self._write_json(HTTPStatus.OK, response)
        except MCPError as exc:
            if request_id is None:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "jsonrpc": "2.0",
                        "error": {"code": exc.code, "message": exc.message},
                    },
                )
            else:
                self._write_json(
                    HTTPStatus.OK,
                    _error_response(request_id, exc.code, exc.message),
                )
        except json.JSONDecodeError as exc:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": f"Invalid JSON: {exc}"},
                },
            )
        except Exception as exc:
            if request_id is None:
                self._write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32603,
                            "message": f"Internal server error: {exc}",
                        },
                    },
                )
            else:
                self._write_json(
                    HTTPStatus.OK,
                    _error_response(
                        request_id, -32603, f"Internal server error: {exc}"
                    ),
                )

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("[mcp-http] " + format % args + "\n")
        sys.stderr.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenSCAD MCP HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), MCPHttpHandler)
    sys.stderr.write(f"Starting openscad MCP HTTP server on {args.host}:{args.port}\n")
    sys.stderr.flush()
    server.serve_forever()


if __name__ == "__main__":
    main()

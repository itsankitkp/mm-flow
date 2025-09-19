# oauth_callback_server.py
# Requires only Python stdlib

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs


class _SimpleCallbackHandler(BaseHTTPRequestHandler):
    """Internal handler; stores parsed query on the HTTPServer instance as .query"""

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != self.server.expected_path:
            self.send_response(404)
            self.end_headers()
            return

        params = parse_qs(parsed.query)
        # normalize: convert single-element lists to values
        normalized = {k: v if len(v) != 1 else v[0] for k, v in params.items()}
        # store results on the server object for the controlling thread
        self.server.query = normalized

        # Simple HTML response for the user's browser (utf-8 safe)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        html = "<html><body><h2>Authorization complete — you may close this window.</h2></body></html>"
        self.wfile.write(html.encode("utf-8"))

        # Shutdown server cleanly from a background thread
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, format, *args):
        # Silence default logging (override if you want logs)
        return


class OAuthCallbackServer:
    """
    Minimal fixed-port loopback server that waits for an OAuth redirect and hands the parsed
    query parameters to a provided consent_handler callable.

    Usage:
        server = OAuthCallbackServer(host="127.0.0.1", port=8080, path="/callback")
        result = server.grant_consent(connector.consent_handler, timeout=120, expected_state="abc123")
    """

    def __init__(
        self, host: str = "127.0.0.1", port: int = 8080, path: str = "/callback"
    ):
        self.host = host
        self.port = port
        self.path = path

    def _make_server(self):
        httpd = HTTPServer((self.host, self.port), _SimpleCallbackHandler)
        httpd.expected_path = self.path
        httpd.query = None
        return httpd

    @property
    def redirect_uri(self) -> str:
        """Computed redirect URI (readable before server.start)."""
        return f"http://{self.host}:{self.port}{self.path}"

    def grant_consent(
        self, consent_handler, timeout: int = 120, expected_state: str = None
    ):
        """
        Start the server and wait for one redirect to `path`. When received, call:
            consent_handler(parsed_params: dict) -> any

        - consent_handler must be a callable accepting single dict arg (parsed query params).
        - timeout: seconds to wait for the callback before raising TimeoutError.
        - expected_state: optional; if provided, server will validate that the 'state' param equals it.
        Returns:
            The return value of consent_handler(...) if not None, otherwise the parsed params dict.
        Raises:
            TimeoutError if no callback in time.
            ValueError if state mismatch (when expected_state provided).
            TypeError if consent_handler isn't callable.
        """
        if not callable(consent_handler):
            raise TypeError(
                "consent_handler must be callable and accept one argument (params dict)"
            )

        server = self._make_server()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            start = time.time()
            while time.time() - start < timeout:
                if getattr(server, "query", None) is not None:
                    break
                time.sleep(0.2)
            else:
                raise TimeoutError(
                    f"No callback received at http://{self.host}:{self.port}{self.path} within {timeout}s"
                )

            params = server.query  # normalized dict (single values are plain strings)

            if expected_state is not None:
                received_state = params.get("state")
                if received_state != expected_state:
                    raise ValueError("State mismatch (possible CSRF)")

            # call connector's handler; let it do code->token exchange, validation, storage, etc.

            handler_result = consent_handler(params)

            return handler_result if handler_result is not None else params

        finally:
            try:
                server.shutdown()
            except Exception:
                pass
            try:
                thread.join(timeout=1)
            except Exception:
                pass

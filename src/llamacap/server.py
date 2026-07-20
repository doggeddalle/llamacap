from __future__ import annotations

import logging
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from llamacap.errors import LlamacapError
from llamacap.profiles import Profile

logger = logging.getLogger("llamacap")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _tail(path: Path, max_chars: int = 2000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


class LlamaServer:
    """Runs a single llama-server process for the lifetime of a batch, so the
    GGUF/mmproj model loads into memory once instead of reloading per image."""

    def __init__(self, binary: Path, profile: Profile, startup_timeout_seconds: int):
        self._binary = binary
        self._profile = profile
        self._startup_timeout_seconds = startup_timeout_seconds
        self._process: subprocess.Popen | None = None
        self._port: int | None = None
        self._log_path: Path | None = None
        self._keep_log = False

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def log_tail(self, max_chars: int = 2000) -> str:
        if self._log_path is None:
            return ""
        return _tail(self._log_path, max_chars)

    def start(self) -> None:
        gen = self._profile.generation
        self._port = _free_port()

        log_fd, log_name = tempfile.mkstemp(prefix="llamacap_server_", suffix=".log")
        self._log_path = Path(log_name)

        argv = [
            str(self._binary),
            "-m", str(self._profile.gguf_path),
            "--mmproj", str(self._profile.mmproj_path),
            "-ngl", str(gen.ngl),
            "-c", str(gen.ctx_size),
            "--image-min-tokens", str(gen.image_min_tokens),
            "--host", "127.0.0.1",
            "--port", str(self._port),
        ]
        if gen.no_warmup:
            argv.append("--no-warmup")
        argv.extend(gen.extra_args)

        logger.debug("server argv: %s", argv)
        logger.info("Starting llama-server on 127.0.0.1:%d (log: %s)", self._port, self._log_path)

        with open(log_fd, "w", encoding="utf-8") as log_file:
            self._process = subprocess.Popen(argv, stdout=log_file, stderr=subprocess.STDOUT)

        self._wait_until_ready()
        logger.info("llama-server ready.")

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self._startup_timeout_seconds
        health_url = f"{self.base_url}/health"
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                self._keep_log = True
                raise LlamacapError(
                    f"llama-server exited during startup (code {self._process.returncode}).\n"
                    f"Log tail:\n{_tail(self._log_path)}"
                )
            try:
                with urllib.request.urlopen(health_url, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.3)

        self._keep_log = True
        self.stop()
        raise LlamacapError(
            f"llama-server did not become ready within {self._startup_timeout_seconds}s.\n"
            f"Log tail:\n{_tail(self._log_path)}"
        )

    def stop(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    try:
                        self._process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        logger.warning("llama-server did not exit after kill")
            else:
                # Died on its own — keep the log around for debugging.
                self._keep_log = True
        self._process = None

        if self._log_path is not None and not self._keep_log:
            try:
                self._log_path.unlink()
            except OSError:
                pass
            self._log_path = None

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

#!/usr/bin/env python3
"""
GPUDB Dashboard Server
──────────────────────
Zero-dependency Python server that:
  1. Serves the dashboard files on http://localhost:8080
  2. Watches benchmark_stream.jsonl for new lines
  3. Pushes updates to the dashboard via Server-Sent Events (SSE)

Usage:
    cd dashboard
    python serve.py

Then open http://localhost:8080 in your browser and run gpudb_report.exe.
"""

import http.server
import json
import os
import sys
import threading
import time
from pathlib import Path

PORT = 8080
JSONL_FILE = "benchmark_stream.jsonl"

# Track connected SSE clients
sse_clients = []
sse_lock = threading.Lock()


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler with SSE endpoint."""

    def do_GET(self):
        if self.path == "/stream":
            self.handle_sse()
        else:
            super().do_GET()

    def handle_sse(self):
        """Server-Sent Events endpoint."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # Register this client
        with sse_lock:
            sse_clients.append(self.wfile)

        print(f"[SSE] Client connected ({len(sse_clients)} total)")

        # Keep the connection alive
        try:
            while True:
                time.sleep(1)
                # Send keepalive comment
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with sse_lock:
                if self.wfile in sse_clients:
                    sse_clients.remove(self.wfile)
            print(f"[SSE] Client disconnected ({len(sse_clients)} total)")

    def log_message(self, format, *args):
        # Suppress noisy request logs except errors
        if args and "404" in str(args[0]):
            super().log_message(format, *args)


def broadcast_sse(data: str):
    """Send data to all connected SSE clients."""
    message = f"data: {data}\n\n"
    encoded = message.encode("utf-8")

    with sse_lock:
        dead = []
        for client in sse_clients:
            try:
                client.write(encoded)
                client.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                dead.append(client)
        for d in dead:
            sse_clients.remove(d)


def watch_jsonl():
    """Watch the JSONL file for new lines and broadcast them via SSE."""
    last_pos = 0
    last_size = 0

    print(f"[Watcher] Monitoring {JSONL_FILE} for new benchmark results...")

    while True:
        try:
            if os.path.exists(JSONL_FILE):
                current_size = os.path.getsize(JSONL_FILE)
                if current_size > last_size:
                    with open(JSONL_FILE, "r") as f:
                        f.seek(last_pos)
                        new_lines = f.readlines()
                        last_pos = f.tell()
                    last_size = current_size

                    for line in new_lines:
                        line = line.strip()
                        if line:
                            broadcast_sse(line)
                            print(f"[Watcher] Broadcast: {line[:80]}...")
                elif current_size < last_size:
                    # File was truncated (new run started)
                    last_pos = 0
                    last_size = 0
        except Exception as e:
            print(f"[Watcher] Error: {e}")

        time.sleep(0.2)  # Poll every 200ms


def main():
    # Change to the directory containing this script
    os.chdir(Path(__file__).parent)

    # Start the file watcher thread
    watcher = threading.Thread(target=watch_jsonl, daemon=True)
    watcher.start()

    # Start the HTTP server
    handler = DashboardHandler
    server = http.server.HTTPServer(("", PORT), handler)

    print(f"""
  ╔════════════════════════════════════════════════════════╗
  ║  GPUDB Dashboard Server                               ║
  ╠════════════════════════════════════════════════════════╣
  ║                                                        ║
  ║  Dashboard:  http://localhost:{PORT}                    ║
  ║  SSE Stream: http://localhost:{PORT}/stream             ║
  ║                                                        ║
  ║  Watching:   {JSONL_FILE:<40s} ║
  ║                                                        ║
  ║  Press Ctrl+C to stop                                  ║
  ╚════════════════════════════════════════════════════════╝
    """)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()

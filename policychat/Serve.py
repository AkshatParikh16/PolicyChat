#!/usr/bin/env python3
"""
PolicyChat Frontend Server
Run this to serve the UI at http://localhost:3000
"""
import http.server
import socketserver
import os

PORT = 3000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def log_message(self, format, *args):
        print(f"  -> {args[0]} {args[1]}")

print(f"""
PolicyChat UI Server
Open: http://localhost:{PORT}
""")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
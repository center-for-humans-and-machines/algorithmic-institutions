"""A stand-in for an agent harness, to test noinet.py as its shell.

Starts a local HTTP server on 127.0.0.1 (the "model server"), one on a Unix
socket and one on an abstract Unix socket, talks HTTP to the TCP one itself,
then runs its "shell tool" through noinet.py: net_probe.py (every socket
route, with an open TCP socket inherited from the parent) and a curl. Then it
talks to its server again. Standard library only.

    python3 harness_standin.py <folder for the Unix socket>
"""

import http.server
import os
import socket
import socketserver
import subprocess
import sys
import threading
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    def get_request(self):
        req, _ = super().get_request()
        return req, ("unix", 0)


def serve(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    folder = sys.argv[1]
    tcp = serve(http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler))
    port = tcp.server_address[1]
    path = os.path.join(folder, "model.sock")
    if os.path.exists(path):
        os.remove(path)
    serve(UnixServer(path, Handler))
    abstract = f"pf-noinet-{os.getpid()}"
    serve(UnixServer("\0" + abstract, Handler))

    url = f"http://127.0.0.1:{port}/"
    print("parent before:", urllib.request.urlopen(url, timeout=5).status)

    def probe(filtered):
        """net_probe.py with a fresh TCP socket to the server, opened here and
        inherited by the child, which sends its own request over it."""
        held = socket.create_connection(("127.0.0.1", port))
        cmd = [
            sys.executable,
            os.path.join(HERE, "net_probe.py"),
            "--http-port",
            str(port),
            "--unix",
            path,
            "--abstract",
            abstract,
            "--inherited-fd",
            str(held.fileno()),
        ]
        if filtered:
            cmd = [sys.executable, os.path.join(HERE, "noinet.py")] + cmd
        subprocess.run(cmd, pass_fds=[held.fileno()], check=False)
        held.close()

    print("-- the same command without the filter (baseline):", flush=True)
    probe(filtered=False)
    print("-- the shell tool's command, through noinet.py:", flush=True)
    probe(filtered=True)
    shell = [
        sys.executable,
        os.path.join(HERE, "noinet.py"),
        "/bin/sh",
        "-c",
        f"curl -s -m 5 -o /dev/null -w 'curl {url}: HTTP %{{http_code}}\\n' {url}"
        ' || echo "curl ' + url + ': failed ($?)"',
    ]
    subprocess.run(shell, check=False)
    print("parent after:", urllib.request.urlopen(url, timeout=5).status)


if __name__ == "__main__":
    main()

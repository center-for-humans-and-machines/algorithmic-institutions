#!/bin/bash
# Apptainer isolation checks for an agent sandbox on DAIS, as an unprivileged
# user. Light enough for a login node; run it inside a job too, where the
# cgroup options behave differently. Writes nothing outside $OUT.
#
#   bash infra/dais/iso_check.sh [image] 2>&1 | tee iso_check.$(hostname).txt
#
# The version run on 2026-11-27 (login node, output not saved) differed: it had
# no fork count for --pids-limit and no seccomp test. This version has not been
# run yet.

set -u
module load apptainer/1.5.2 >/dev/null 2>&1
S="${1:-/dais/fs/scratch/bsup/container/vllm.sif}"
OUT="${OUT:-/dais/fs/scratch/$USER/policy-finder/isotest}"
mkdir -p "$OUT/sock" "$OUT/ro/rw"
q() { grep -v gocryptfs; }
echo "host $(hostname), apptainer $(apptainer --version), image $S"

echo "== 1. --containall --net --network none"
apptainer exec --containall --net --network none "$S" true 2>&1 | q
echo "rc=${PIPESTATUS[0]} max_net_namespaces=$(cat /proc/sys/user/max_net_namespaces)"

echo "== 1b. without a network namespace: host loopback from inside"
python3 -m http.server 18765 --bind 127.0.0.1 >/dev/null 2>&1 &
HP=$!
sleep 1
apptainer exec --containall "$S" curl -s -m 5 -o /dev/null \
  -w "host 127.0.0.1:18765 -> %{http_code}\n" http://127.0.0.1:18765/ 2>&1 | q
kill $HP

echo "== 2. HTTP on a Unix socket outside, client inside"
rm -f "$OUT/sock/model.sock"
python3 - "$OUT/sock/model.sock" <<'PY' &
import http.server, socketserver, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        b = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def log_message(self, *a):
        pass
class S(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    def get_request(self):
        r, _ = super().get_request()
        return r, ("unix", 0)
S(sys.argv[1], H).serve_forever()
PY
UP=$!
sleep 1
apptainer exec --containall --bind "$OUT/sock:/sock" "$S" \
  curl -s -m 5 --unix-socket /sock/model.sock http://x/v1/models -w " -> %{http_code}\n" 2>&1 | q
echo "-- a relay inside: 127.0.0.1:18766 -> /sock/model.sock"
apptainer exec --containall --bind "$OUT/sock:/sock" "$S" sh -c '
python3 - <<"PY" &
import socket, threading
def pipe(a, b):
    try:
        while (d := a.recv(65536)):
            b.sendall(d)
    finally:
        b.close()
l = socket.socket()
l.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
l.bind(("127.0.0.1", 18766))
l.listen()
while True:
    c, _ = l.accept()
    u = socket.socket(socket.AF_UNIX)
    u.connect("/sock/model.sock")
    threading.Thread(target=pipe, args=(c, u), daemon=True).start()
    threading.Thread(target=pipe, args=(u, c), daemon=True).start()
PY
sleep 1
curl -s -m 5 http://127.0.0.1:18766/v1/models -w " -> %{http_code}\n"
kill $!' 2>&1 | q

echo "== 2b. seccomp profile denying AF_INET/AF_INET6 sockets, AF_UNIX allowed"
cat > "$OUT/no_inet.json" <<'JSON'
{"defaultAction": "SCMP_ACT_ALLOW",
 "syscalls": [
  {"names": ["socket"], "action": "SCMP_ACT_ERRNO", "args": [{"index": 0, "value": 2, "op": "SCMP_CMP_EQ"}]},
  {"names": ["socket"], "action": "SCMP_ACT_ERRNO", "args": [{"index": 0, "value": 10, "op": "SCMP_CMP_EQ"}]}
 ]}
JSON
apptainer exec --containall --security "seccomp:$OUT/no_inet.json" \
  --bind "$OUT/sock:/sock" "$S" sh -c '
curl -s -m 5 -o /dev/null -w "inet: %{http_code}\n" https://huggingface.co || echo "inet: refused ($?)"
curl -s -m 5 --unix-socket /sock/model.sock http://x/v1/models -w " unix -> %{http_code}\n"' 2>&1 | q
echo "rc=${PIPESTATUS[0]}"
kill $UP

echo "== 3. binds"
echo hi > "$OUT/ro/f"
rm -f "$OUT/ro/rw/y"
apptainer exec --containall --bind "$OUT/ro:/work:ro" --bind "$OUT/ro/rw:/work/rw" "$S" sh -c '
touch /work/x 2>/dev/null && echo "ro bind: writable (unexpected)" || echo "ro bind: read-only"
touch /work/rw/y && echo "rw sub-bind: writable"
ls /dais/fs/scratch >/dev/null 2>&1 && echo "scratch: visible" || echo "scratch: not visible"
echo "HOME=$HOME entries=$(ls -A "$HOME" | wc -l)"
ls /u >/dev/null 2>&1 && echo "/u: visible ($(ls /u | wc -l) entries)" || echo "/u: not visible"
touch /usr/zz 2>/dev/null && echo "image root: writable (unexpected)" || echo "image root: read-only"
touch /tmp/zz && echo "/tmp: writable"' 2>&1 | q
ls "$OUT/ro/rw"

echo "== 4. resource limits"
for f in "--cpus 2" "--memory 2G"; do
  if err=$(apptainer exec $f "$S" true 2>&1); then echo "$f: works"
  else echo "$f: refused: $(echo "$err" | grep -E 'ERROR|FATAL' | head -1)"; fi
done
echo "-- --pids-limit 64: start 200 sleepers, count the ones that started"
apptainer exec --pids-limit 64 "$S" sh -c '
n=0; for i in $(seq 200); do sleep 5 2>/dev/null & [ $? = 0 ] && n=$((n + 1)); done
echo "started $n of 200, running now: $(jobs -p | wc -l)"' 2>&1 | q | tail -2
echo "-- the same without a limit"
apptainer exec "$S" sh -c '
n=0; for i in $(seq 200); do sleep 5 2>/dev/null & [ $? = 0 ] && n=$((n + 1)); done
echo "started $n of 200, running now: $(jobs -p | wc -l)"' 2>&1 | q | tail -2

echo "== 5. runtimes"
for c in podman docker nerdctl buildah slirp4netns pasta; do
  command -v "$c" >/dev/null && echo "$c: present" || echo "$c: absent"
done
grep -c "^$USER:" /etc/subuid 2>/dev/null | sed 's/^/subuid entries: /'

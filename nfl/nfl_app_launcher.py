"""Start/stop only this project's loopback-only Streamlit server."""
from pathlib import Path
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent
STATE = ROOT / '.nfl_app_server.json'
URL = 'http://127.0.0.1:8501'


def own_server():
    if not STATE.exists():
        return None
    try:
        pid = int(json.loads(STATE.read_text())['pid'])
        command = subprocess.check_output(['ps', '-p', str(pid), '-o', 'command='], text=True).strip()
        return pid if str(ROOT / 'nfl_app.py') in command and 'streamlit' in command else None
    except (ValueError, KeyError, subprocess.CalledProcessError):
        return None


def launch_nfl_app(open_browser=True):
    python = ROOT.parent / '.venv/bin/python'
    if not python.exists():
        raise RuntimeError('Project .venv is missing.')
    subprocess.run([str(python), '-c', 'import streamlit'], check=True)
    if own_server() is None:
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', 8501)) == 0:
                raise RuntimeError('Port 8501 is used by another server. Stop that server before launching this app.')
        with (ROOT / '.nfl_app_server.log').open('a') as log:
            process = subprocess.Popen([
                str(python), '-m', 'streamlit', 'run', str(ROOT / 'nfl_app.py'),
                '--server.address=127.0.0.1', '--server.port=8501',
                '--server.headless=true', '--browser.gatherUsageStats=false',
            ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        STATE.write_text(json.dumps({'pid': process.pid}))
    for _ in range(60):
        try:
            with urllib.request.urlopen(URL + '/_stcore/health', timeout=1) as response:
                if response.status == 200:
                    if open_browser:
                        webbrowser.open(URL)
                    return URL
        except OSError:
            time.sleep(.2)
    raise RuntimeError('Server did not become ready. Check .nfl_app_server.log.')


def stop_nfl_app():
    pid = own_server()
    if pid is not None:
        os.kill(pid, signal.SIGTERM)
        return 'NFL app stopped.'
    return 'The NFL app is not running.'


if __name__ == '__main__':
    print(launch_nfl_app())

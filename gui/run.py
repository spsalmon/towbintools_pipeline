import argparse
import os
import signal
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Launch the annotation GUI")
    parser.add_argument("--filemap", type=str, default=None)
    parser.add_argument("--no-annotated", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--host", type=str, default="127.0.0.1")

    args = parser.parse_args()

    env = os.environ.copy()
    if args.filemap:
        env["FILEMAP_PATH"] = args.filemap
    env["OPEN_ANNOTATED"] = "0" if args.no_annotated else "1"
    env["RECOMPUTE_FEATURES"] = "1" if args.recompute else "0"

    app_dir = os.path.dirname(os.path.abspath(__file__))

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "shiny",
            "run",
            "app.py",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ],
        cwd=app_dir,
        env=env,
        start_new_session=True,
    )

    def _sigint_handler(sig, frame):
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()

    signal.signal(signal.SIGINT, _sigint_handler)
    proc.wait()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

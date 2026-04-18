import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def build_frontend():
    frontend_dir = BASE_DIR / "frontend"
    if not frontend_dir.exists():
        print("WARN: frontend/ directory not found -- skipping build")
        return
    print("Building frontend...")
    node_modules = frontend_dir / "node_modules"
    if not node_modules.exists():
        print("   Installing npm packages...")
        subprocess.run(["npm", "install"], cwd=str(frontend_dir), check=True, shell=True)
    subprocess.run(["npm", "run", "build"], cwd=str(frontend_dir), check=True, shell=True)
    print("OK: Frontend built -> frontend/dist/")


def is_port_free(host: str, port: int) -> bool:
    """
    Return True if the port is available to bind.

    NOTE: do NOT set SO_REUSEADDR here — on Windows that flag allows binding
    to an already-occupied port, making the check always return True.
    We use a connect() probe instead of a bind() so we never actually
    reserve the port ourselves.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        try:
            s.connect((host if host != "0.0.0.0" else "127.0.0.1", port))
            # Connection succeeded → something is already listening → port is busy
            return False
        except (ConnectionRefusedError, socket.timeout, OSError):
            # Nothing listening → port is free
            return True


def find_free_port(host: str, preferred: int, attempts: int = 3) -> int:
    """
    Try `preferred` first, then preferred+1, preferred+2, …
    Raises RuntimeError if none are free after `attempts` tries.
    """
    for offset in range(attempts):
        port = preferred + offset
        if is_port_free(host, port):
            return port
        print(f"  Port {port} is in use, trying {port + 1}...")
    raise RuntimeError(
        f"Could not find a free port after {attempts} attempts "
        f"(tried {preferred}–{preferred + attempts - 1}).\n"
        f"Kill the process using port {preferred} or pass --port <number>."
    )


def main():
    parser = argparse.ArgumentParser(description="Start SENTINEL")
    parser.add_argument("--build", action="store_true", help="Build frontend before starting")
    parser.add_argument("--dev",   action="store_true", help="Enable auto-reload (development)")
    parser.add_argument("--host",  default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port",  type=int, default=int(os.getenv("PORT", 8000)))
    args = parser.parse_args()

    if args.build:
        build_frontend()

    # Ensure data directory exists
    (BASE_DIR / "data").mkdir(exist_ok=True)

    # Auto-find a free port (tries preferred, preferred+1, preferred+2)
    try:
        port = find_free_port(args.host, args.port)
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    if port != args.port:
        print(f"  Using port {port} instead of {args.port}.")

    print(f"\n{'='*60}")
    print(f"  SENTINEL Analytics")
    print(f"  URL:  http://localhost:{port}")
    print(f"  Docs: http://localhost:{port}/api/docs")
    print(f"{'='*60}\n")

    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=port,
        reload=args.dev,
        log_level="info",
    )


if __name__ == "__main__":
    main()

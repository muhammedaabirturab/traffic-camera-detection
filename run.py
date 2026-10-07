"""Start the TrafficGuard AI server (API + built dashboard).

    python run.py                 # http://127.0.0.1:8000
    python run.py --port 8080 --reload
"""

import argparse

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the TrafficGuard AI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")
    args = parser.parse_args()
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)

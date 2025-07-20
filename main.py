import argparse
import asyncio
import uvicorn
from routes import app

def run_server(host="127.0.0.1", port=11435, reload=True):
    uvicorn.run("main:app", host=host, port=port, reload=reload)

def main():
    parser = argparse.ArgumentParser(description="LangGraph Agent Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=11435, help="Server port")
    parser.add_argument("--no-reload", action="store_true", help="Disable reload")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, reload=not args.no_reload)

if __name__ == "__main__":
    main()

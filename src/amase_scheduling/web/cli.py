"""Launch the AMASE scheduling web UI."""

import argparse
import threading
import webbrowser


def main() -> None:
    from amase_scheduling._warnings import suppress_future_date_warnings
    suppress_future_date_warnings()

    parser = argparse.ArgumentParser(
        prog="amase-web", description="Start the AMASE-P scheduling web UI."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically.",
    )
    args = parser.parse_args()
    import uvicorn  # lazy: importing amase_scheduling must never require fastapi/uvicorn

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(
        f"AMASE-P scheduling web UI at {url} (Exit button in the page stops the server)"
    )
    uvicorn.run("amase_scheduling.web.main:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()

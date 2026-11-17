"""
PriviScan model preloading.

The OCR model (PaddleOCR) and the AI model (GLiNER) take a long time to load.
Loading them inside the server startup made the web page wait for both models,
one after the other, every time the server started (and again after every
--reload restart).

Now the web server is ready immediately and the two models load IN THE
BACKGROUND, at the same time. An upload that arrives while a model is still
loading simply waits for it (the model loaders are thread-safe).

Environment variable
--------------------
PRIVISCAN_PRELOAD        "background" (default)  server ready at once
                         "blocking"               wait until both models are loaded
                         "off"                    load on the first upload only
PRIVISCAN_PARALLEL_LOAD  "1" (default) load OCR and GLiNER at the same time,
                         "0" load them one after the other
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Dict, List, Tuple

STATUS: Dict[str, str] = {"ocr": "not started", "gliner": "not started"}

_started = False
_lock = threading.Lock()


def _run(name: str, function: Callable[[], None]) -> None:
    STATUS[name] = "loading"
    started = time.perf_counter()
    try:
        function()
        STATUS[name] = "ready"
    except Exception as exc:  # one model failing must not stop the other
        STATUS[name] = f"failed: {exc}"
        print(f"[PriviScan] {name} could not be preloaded: {exc}")
    print(f"[PriviScan] {name}: {STATUS[name]} ({time.perf_counter() - started:.1f}s)")


def start_preload() -> List[threading.Thread]:
    """Start loading the models. Returns the worker threads."""
    global _started

    mode = os.environ.get("PRIVISCAN_PRELOAD", "background").strip().lower()

    with _lock:
        if _started or mode == "off":
            return []
        _started = True

    from backend.ocr.detector import warm_up as warm_up_ocr
    from backend.detection.ai_detector import warm_up as warm_up_ai

    jobs: List[Tuple[str, Callable[[], None]]] = [
        ("ocr", warm_up_ocr),
        ("gliner", warm_up_ai),
    ]

    if os.environ.get("PRIVISCAN_PARALLEL_LOAD", "1") == "0":
        def in_sequence() -> None:
            for name, function in jobs:
                _run(name, function)

        threads = [threading.Thread(target=in_sequence, daemon=True, name="priviscan-preload")]
    else:
        threads = [
            threading.Thread(target=_run, args=job, daemon=True, name=f"priviscan-preload-{job[0]}")
            for job in jobs
        ]

    for thread in threads:
        thread.start()

    if mode == "blocking":
        for thread in threads:
            thread.join()
    else:
        print("[PriviScan] Server is ready. OCR and AI models are loading in the background...")

    return threads

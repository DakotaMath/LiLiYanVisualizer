"""Browser version of Swirl Explorer: the Python side of the static web page (site/).

The page runs this package in the visitor's browser with Pyodide (CPython in WebAssembly) and calls
``api.call(name, json_args)``; ``rows`` serves the Σ-map worker pool.  Only numpy, scipy and
matplotlib are needed here: no Qt, no VTK (tests/test_layout.py and tests/test_web_api.py check it).
See docs/web_version.md.
"""

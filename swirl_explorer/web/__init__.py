"""Browser version of Swirl Explorer (trame), built on the same core as the desktop app.

Layout of the package:

    core/  viz/  io/    shared by both front ends; no Qt imports (checked by tests/test_layout.py)
    ui/                 desktop front end (PySide6 + pyvistaqt)
    web/                browser front end (trame + PyVista), deployed to a Hugging Face Space

Install with ``pip install -r requirements-web.txt``.  Nothing in ``web/`` may be imported
by ``ui/`` or by the shared packages, and the Windows build leaves ``web/`` out.
"""

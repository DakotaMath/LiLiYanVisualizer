"""PyInstaller entry script for the Windows build (SwirlExplorer.exe)."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()     # must run first: Σ-map worker processes start this exe again
    from swirl_explorer.__main__ import main
    sys.exit(main())

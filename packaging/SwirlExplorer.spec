# PyInstaller spec for the one-folder Windows build.  Built by build_windows.ps1:
#     pyinstaller --noconfirm --clean packaging/SwirlExplorer.spec
# Output: dist/SwirlExplorer/SwirlExplorer.exe (+ _internal/ with Python, Qt, VTK, ...)
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
sys.path.insert(0, ROOT)
from swirl_explorer import __version__  # noqa: E402

# the VTK modules PyVista loads at run time (rendering back-ends are picked by the object factory)
hidden = [
    "vtkmodules.vtkRenderingOpenGL2", "vtkmodules.vtkInteractionStyle", "vtkmodules.vtkRenderingFreeType",
    "vtkmodules.vtkRenderingContextOpenGL2", "vtkmodules.vtkRenderingVolumeOpenGL2",
    "vtkmodules.vtkRenderingUI", "vtkmodules.vtkInteractionWidgets", "vtkmodules.vtkRenderingAnnotation",
    "vtkmodules.vtkFiltersFlowPaths", "vtkmodules.vtkFiltersModeling", "vtkmodules.vtkFiltersSources",
    "vtkmodules.util.numpy_support", "vtkmodules.qt.QVTKRenderWindowInteractor",
    "pyvistaqt", "scipy.integrate._ode", "scipy.integrate._dop",
    "matplotlib.backends.backend_qtagg", "matplotlib.backends.backend_agg",
    "imageio.plugins.ffmpeg", "imageio_ffmpeg",
] + collect_submodules("swirl_explorer", filter=lambda name: not name.startswith("swirl_explorer.web"))

datas = (collect_data_files("imageio_ffmpeg", include_py_files=False)      # the ffmpeg binary
         + collect_data_files("pyvista", includes=["**/*.json", "**/*.txt"])
         + [(os.path.join(ROOT, "README.md"), "."),
            (os.path.join(ROOT, "packaging", "swirl.png"), ".")])

# large parts of Qt and the scientific stack the app never touches
excludes = [
    "swirl_explorer.web",                       # the browser front end is not part of the desktop app
    "tkinter", "IPython", "jupyter", "notebook", "ipywidgets", "trame", "trame_vtk", "trame_vuetify",
    "pytest", "sphinx", "pygame", "PIL.ImageQt", "pandas", "sympy", "numba", "PyQt5", "PyQt6",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick", "PySide6.QtWebChannel",
    "PySide6.QtPdf", "PySide6.QtPdfWidgets", "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DExtras",
    "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQml", "PySide6.QtMultimedia", "PySide6.QtCharts",
    "PySide6.QtDataVisualization", "PySide6.QtGraphs", "PySide6.QtBluetooth", "PySide6.QtPositioning",
    "PySide6.QtLocation", "PySide6.QtSensors", "PySide6.QtSerialPort", "PySide6.QtTextToSpeech",
    "PySide6.QtWebSockets", "PySide6.QtNetworkAuth", "PySide6.QtSql", "PySide6.QtDesigner", "PySide6.QtHelp",
    "PySide6.QtSpatialAudio", "PySide6.QtRemoteObjects", "PySide6.QtScxml", "PySide6.QtStateMachine",
]

a = Analysis(
    [os.path.join(ROOT, "packaging", "launch.py")],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

# Windows version resource (shows in Explorer > Properties > Details)
from PyInstaller.utils.win32.versioninfo import (VSVersionInfo, FixedFileInfo, StringFileInfo,  # noqa: E402
                                                 StringTable, StringStruct, VarFileInfo, VarStruct)
nums = tuple(int(x) for x in (__version__.split(".") + ["0", "0", "0"])[:4])
version_info = VSVersionInfo(
    ffi=FixedFileInfo(filevers=nums, prodvers=nums),
    kids=[StringFileInfo([StringTable("040904B0", [
        StringStruct("FileDescription", "Swirl Explorer"),
        StringStruct("ProductName", "Swirl Explorer"),
        StringStruct("FileVersion", __version__),
        StringStruct("ProductVersion", __version__),
        StringStruct("OriginalFilename", "SwirlExplorer.exe"),
    ])]), VarFileInfo([VarStruct("Translation", [0x0409, 1200])])],
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SwirlExplorer",
    icon=os.path.join(ROOT, "packaging", "swirl.ico"),
    version=version_info,
    console=False,               # a windowed app; --selftest writes its report to a file
    upx=False,                   # UPX-packed Qt/VTK DLLs trip antivirus scanners
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="SwirlExplorer")

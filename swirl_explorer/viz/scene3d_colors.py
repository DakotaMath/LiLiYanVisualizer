"""Colours and names used by the 3D view and the Σ-map (no PyVista, so the browser build can use them)."""

STRATUM_COLORS = {"G": "#2a78d6", "R": "#e34948", "L1": "#eda100", "L2": "#8f3fd1",
                  "R0": "#e87ba4", "unsafe": "#8a8986", "blowup": "#8a8986", "?": "#8a8986"}
COLOR_BY = {"speed": "|u| (log scale)", "u_phi": "u_φ (swirl velocity)", "helicity": "helicity u·ω",
            "pressure": "pressure p"}
VTK_TITLES = {"speed": "speed (log scale)", "u_phi": "swirl velocity",
              "helicity": "helicity", "pressure": "pressure"}
ISO_FIELDS = {"speed": "|u|", "swirl": "swirl f / sin θ", "helicity": "helicity u·ω"}

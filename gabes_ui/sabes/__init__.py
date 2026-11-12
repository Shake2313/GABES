"""SABES screen: the optical-table page, its canvas component and drawing spec.

The SABES model (beamline, instruments, layout geometry) stays in `sabes`;
this package only draws it. `page.render(host=...)` is the entry point that
`streamlit_app.py` calls for `?app=sabes`.
"""

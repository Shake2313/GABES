"""Explicit single-beam calculation inside the SABES profiler instrument.

Atomic tables, cell fields and camera navigation are separate cache tiers.
Only the Run action starts material propagation; stale physical inputs never
reuse a displayed result. Apparatus and FWM settings are not mutated here.
"""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import scipy
import streamlit as st

from gabes import single_beam_profile as model
from gabes.beam import peak_intensity
from gabes.optical_frequency import RbOpticalFrequency
from gabes.plot_style import PALETTE, apply_gabes_plot_style
from gabes_ui.plotcard import PLOT_LOCK, render_fig
from gabes_ui.readout import strip_html
from sabes.session_export import json_value
from sabes.single_beam import cell_beam_input, frequency_references

PREFIX = "_sabes_profiler_"
INPUTS = ("Custom single beam", "Cell pump", "Cell seed line")
MODEL_SCOPE = ("One isolated CW line, linear polarization, B=0; local full-OBE Maxwell response. "
               "Other optical drives, thermal lens and transverse atomic transport are absent. "
               "One selected D-line manifold; remote fine-structure contributions are not included.")


def key(name):
    return PREFIX + name


def _preset(sign):
    values = dict(line="D1", reference="rb87_d1_upper_ground_centroid", detuning_mhz=sign*1500.,
        rb85_percent=0., temp_c=100., length_mm=50., reset_rate=5e5,
        power_mw=5.02654824574367, radius_mm=.4, planar=True, curvature_m=1.,
        window_mm=2.4, samples=128, steps=80, order=6, cutoff=6.,
        ceiling_factor=4., max_nodes=129, refine=True)
    for name, value in values.items():
        st.session_state[key(name)] = value
    st.session_state[key("frequency")] = RbOpticalFrequency.from_detuning(
        "Rb87", "D1", sign*1500e6, reference="upper_ground_centroid")


PERSISTED_DEFAULTS = dict(line="D1", reference="rb87_d1_upper_ground_centroid", detuning_mhz=1500.,
    rb85_percent=0., temp_c=100., length_mm=50., reset_rate=5e5,
    power_mw=5.02654824574367, radius_mm=.4, planar=True, curvature_m=1.,
    window_mm=2.4, samples=128, steps=80, order=6, cutoff=6., ceiling_factor=4.,
    max_nodes=129, refine=True, source=INPUTS[0], camera_mm=500., view="Camera profile")


def _seed():
    # Streamlit removes hidden widget keys. The separate snapshot is never a
    # widget, so source switches and leaving the whole instrument retain inputs.
    saved = st.session_state.get(key("saved_inputs"), {})
    if key("frequency") not in st.session_state:
        _preset(1.)
    for name, default in PERSISTED_DEFAULTS.items():
        st.session_state[key(name)] = st.session_state.get(key(name), saved.get(name, default))
    if "seed_offset" in saved:
        st.session_state[key("seed_offset")] = st.session_state.get(key("seed_offset"), saved["seed_offset"])
    st.session_state.setdefault(key("last_source"), st.session_state[key("source")])


def _remember_inputs():
    names = tuple(PERSISTED_DEFAULTS) + ("seed_offset",)
    st.session_state[key("saved_inputs")] = {
        name: st.session_state[key(name)] for name in names if key(name) in st.session_state}


def _reference():
    refs = frequency_references(st.session_state[key("line")])
    return next(ref for ref in refs if ref.key == st.session_state[key("reference")])


def _reference_changed():
    st.session_state[key("detuning_mhz")] = _reference().detuning_mhz(st.session_state[key("frequency")])


def _detuning_changed():
    st.session_state[key("frequency")] = _reference().frequency(st.session_state[key("detuning_mhz")])


def _line_changed():
    line = st.session_state[key("line")]
    st.session_state[key("reference")] = f"rb87_{line.lower()}_upper_ground_centroid"
    _detuning_changed()


def cache_version():
    """Fingerprint the imported numerical dependency sources, not UI display state."""
    from gabes import (angular, atoms, beam, constants, core, doppler, magneto_multilevel,
                       _arrays, magneto_coordinates, magneto_optical, zeeman,
                       nonlinear_response, optical_bloch, optical_frequency, paraxial,
                       psr_conventions, species)
    from gabes.quantum import contracts, reservoirs
    from gabes import psr_full_obe
    modules = (model, angular, atoms, beam, constants, core, doppler, magneto_multilevel,
               _arrays, magneto_coordinates, magneto_optical, zeeman,
               nonlinear_response, optical_bloch, optical_frequency, paraxial,
               psr_conventions, species, contracts, reservoirs, psr_full_obe)
    digest = sha256((np.__version__+scipy.__version__).encode("ascii"))
    for module in modules:
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_table(settings, bound, length, max_nodes, version, _progress=None):
    return model.build_response_table(settings, bound, cell_length_m=length,
                                     max_nodes=max_nodes, progress=_progress)


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_cell(settings, bound, length, max_nodes, power, radius, curvature,
                 window, samples, steps, refine, version, _progress=None):
    response = _cached_table(settings, bound, length, max_nodes, version,
        _progress=(lambda value, text: _progress(.7*value, text)) if _progress else None)
    return model.propagate_cell(response, power_w=power, radius_m=radius,
        curvature_radius_m=curvature, cell_length_m=length, half_width_m=window,
        samples=samples, steps=steps, refine=refine,
        progress=(lambda value, text: _progress(.7+.3*value, text)) if _progress else None)


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_camera(cell_key, distance, version, _cell):
    return model.profile_at_distance(_cell, distance)


def _controls(chain, temperature_c, cell_length_m):
    source = st.selectbox("Input beam", INPUTS, key=key("source"))
    imported = source != INPUTS[0]
    previous_source = st.session_state[key("last_source")]
    if source != previous_source:
        custom_names = ("line", "reference", "detuning_mhz", "frequency", "rb85_percent",
                        "temp_c", "length_mm", "reset_rate", "power_mw", "radius_mm",
                        "planar", "curvature_m")
        if previous_source == INPUTS[0]:
            st.session_state[key("custom_inputs")] = {
                name: st.session_state[key(name)] for name in custom_names}
        elif source == INPUTS[0]:
            for name, value in st.session_state.get(key("custom_inputs"), {}).items():
                st.session_state[key(name)] = value
        st.session_state[key("last_source")] = source
    beam_input = None
    if imported:
        offset = None
        if source == "Cell seed line":
            options = [line.offset_hz for line in chain.seed.lines if line.power_w > 0.]
            labels = {line.offset_hz: f"{line.label or 'Line'} | {line.offset_hz/1e9:+.6f} GHz | {line.power_w*1e6:.4g} \u00b5W"
                      for line in chain.seed.lines}
            if st.session_state.get(key("seed_offset")) not in options:
                st.session_state[key("seed_offset")] = chain.seed_offset_hz
            offset = st.selectbox("Isolated seed component", options, key=key("seed_offset"),
                                  format_func=labels.get)
        beam_input = cell_beam_input(chain, "pump" if source == "Cell pump" else "seed",
                                     cell_length_m, offset_hz=offset)
        st.session_state[key("frequency")] = beam_input.frequency
        st.session_state[key("line")] = beam_input.frequency.line
        refs = frequency_references(beam_input.frequency.line)
        if st.session_state[key("reference")] not in {ref.key for ref in refs}:
            st.session_state[key("reference")] = refs[0].key
        _reference_changed()
        st.session_state[key("temp_c")] = float(temperature_c)
        st.session_state[key("length_mm")] = float(cell_length_m*1e3)
    else:
        red, blue = st.columns(2)
        red.button("Load red reference", key=key("red"), on_click=_preset, args=(-1.,), width="stretch")
        blue.button("Load blue reference", key=key("blue"), on_click=_preset, args=(1.,), width="stretch")
    conditions, numerics = st.columns(2)
    with conditions.popover("Cell & laser", width="stretch"):
        st.selectbox("D line", ("D1", "D2"), key=key("line"),
                     disabled=imported, on_change=_line_changed)
        refs = {ref.key: ref for ref in frequency_references(st.session_state[key("line")])}
        st.selectbox("Frequency reference", tuple(refs), key=key("reference"),
                     format_func=lambda value: refs[value].label, on_change=_reference_changed)
        st.number_input("Detuning (MHz)", key=key("detuning_mhz"), step=10.,
            format="%.6f", disabled=imported, on_change=_detuning_changed,
            help="Laser minus the selected reference. Changing the reference preserves the laser frequency.")
        st.number_input("\u2078\u2075Rb fraction (%)", min_value=0., max_value=100., step=.1,
            key=key("rb85_percent"), help="Independent profiler composition; 0% is pure Rb87, 100% is pure Rb85.")
        st.number_input("Vapour temperature (\u00b0C)", min_value=0., max_value=200., step=1.,
                        key=key("temp_c"), disabled=imported)
        st.number_input("Cell length (mm)", min_value=.01, max_value=1000., step=1.,
                        key=key("length_mm"), disabled=imported)
        st.number_input("Ground reset rate (s\u207b\u00b9)", min_value=1., max_value=1e8, step=10000.,
            key=key("reset_rate"), format="%.4g", help="Fixed effective reset; not inferred from the beam radius.")
        if not imported:
            st.number_input("Input power (mW)", min_value=1e-9, max_value=10000., step=1.,
                            format="%.7g", key=key("power_mw"))
            st.number_input("Entrance radius (mm)", min_value=.005, max_value=10., step=.01,
                            format="%.5f", key=key("radius_mm"))
            st.checkbox("Planar entrance wavefront", key=key("planar"))
            if not st.session_state[key("planar")]:
                st.number_input("Wavefront curvature R (m)", step=.1, key=key("curvature_m"),
                                help="Positive is diverging; zero is invalid.")
    with numerics.popover("Numerical checks", width="stretch"):
        st.checkbox("Check propagation convergence", key=key("refine"),
                    help="Refine transverse grid, propagation step, window and intensity table.")
        st.selectbox("Grid samples per axis", (64, 128, 256, 512), key=key("samples"))
        st.number_input("Window half-width (mm)", min_value=.05, max_value=30., step=.2, key=key("window_mm"))
        st.number_input("Cell steps", min_value=20, max_value=1000, step=20, key=key("steps"))
        st.number_input("Velocity order", min_value=2, max_value=16, step=2, key=key("order"))
        st.number_input("Maxwell cutoff (\u03c3)", min_value=4., max_value=10., step=.5, key=key("cutoff"))
        st.number_input("Intensity ceiling / input peak", min_value=1.1, max_value=64., step=1., key=key("ceiling_factor"))
        st.number_input("Maximum intensity nodes", min_value=17, max_value=1025, step=16, key=key("max_nodes"))
    frequency = st.session_state[key("frequency")]
    length = st.session_state[key("length_mm")]*1e-3
    if beam_input is None:
        power = st.session_state[key("power_mw")]*1e-3
        radius = st.session_state[key("radius_mm")]*1e-3
        curvature = np.inf if st.session_state[key("planar")] else st.session_state[key("curvature_m")]
        metadata = dict(source_label="Custom single beam", reference_plane="cell entrance",
            power_w=power, radius_m=radius, curvature_radius_m=curvature, coherent_m2=1.,
            frequency=frequency.as_dict(), scope="one isolated line; other drives absent")
    else:
        power, radius, curvature = beam_input.power_w, beam_input.radius_m, beam_input.curvature_radius_m
        metadata = beam_input.as_dict()
    metadata = dict(metadata, displayed_frequency_reference=_reference().as_dict(),
                    displayed_detuning_mhz=st.session_state[key("detuning_mhz")])
    settings = model.AtomicSettings(frequency,
        rb85_fraction=st.session_state[key("rb85_percent")]/100.,
        temp_c=st.session_state[key("temp_c")], ground_relaxation_hz=st.session_state[key("reset_rate")],
        velocity_order=st.session_state[key("order")], cutoff_sigma=st.session_state[key("cutoff")])
    bound = peak_intensity(power, radius)*st.session_state[key("ceiling_factor")]
    arguments = (settings, bound, length, st.session_state[key("max_nodes")], power, radius,
        curvature, st.session_state[key("window_mm")]*1e-3, st.session_state[key("samples")],
        st.session_state[key("steps")], st.session_state[key("refine")])
    st.caption(f"{frequency.absolute_hz/1e12:.9f} THz | {frequency.wavelength_m*1e9:.6f} nm | "
               f"{power*1e3:.6g} mW | entrance radius {radius*1e3:.5g} mm | "
               f"Rb85 {100*settings.rb85_fraction:.4g}% | {settings.temp_c:g} \u00b0C | {length*1e3:g} mm")
    if beam_input is not None:
        st.caption("Isolated pre-cell component. Nominal cell-centre mode is propagated back to the entrance; "
                   "actual waist position is unmeasured. Other drives are absent from this calculation.")
    return arguments, metadata


def export_profile(cell, camera, metadata, *, calculation_fingerprint=None):
    """Native arrays and strict JSON conditions/checks from the same saved result."""
    record = dict(model_id=cell["model_id"], source=metadata, conditions=cell["conditions"],
        camera_distance_m=camera["distance_m"], camera_plane="free space after cell exit; no downstream optics",
        checks=camera["checks"], nonlinear_lens=cell["nonlinear_lens"], model_scope=MODEL_SCOPE,
        numerical_fingerprint=calculation_fingerprint,
        runtime=dict(numpy=np.__version__, scipy=scipy.__version__))
    text = json.dumps(json_value(record), ensure_ascii=False, indent=2, allow_nan=False)
    arrays = dict(x_m=cell["grid"].x_m, y_m=cell["grid"].y_m, metadata_json=np.array(text))
    for name, value in cell["paths"].items():
        arrays[name+"_exit_field"] = value.final_field
        arrays[name+"_camera_field"] = camera["paths"][name].final_field
        for field, data in camera["combined"][name].items():
            arrays[name+"_"+field] = data
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    return text, stream.getvalue()


def _plots(cell, camera, view):
    import matplotlib.pyplot as plt
    labels = dict(full="Full response", absorption_only="Absorption only", vacuum="Vacuum")
    colours = dict(full=PALETTE["cyan"], absorption_only=PALETTE["warm"], vacuum=PALETTE["muted"])
    with PLOT_LOCK:
        figure, axis = plt.subplots(figsize=(7.2, 3.8))
        try:
            if view == "Camera profile":
                for name, path in camera["paths"].items():
                    axis.plot(path.x_m*1e3, path.centerline_intensity_w_m2[-1]/1e4,
                              label=labels[name], color=colours[name])
                axis.set(xlabel="Camera x (mm)", ylabel="Intensity (W/cm^2)")
                axis.legend()
            elif view == "Beam radius":
                for name, path in camera["combined"].items():
                    radius = np.where(path["power_w"]/cell["incident_power_w"] >= model.DARK_TRANSMISSION,
                                      path["radius_m"], np.nan)
                    axis.plot(path["z_m"]*1e3, radius*1e3, label=labels[name], color=colours[name])
                axis.axvline(cell["cell_length_m"]*1e3, color=PALETTE["border"], linestyle="--")
                axis.set(xlabel="Distance from cell entrance (mm)", ylabel="Second-moment radius (mm)")
                axis.legend()
            else:
                path = camera["paths"]["full"]
                image = axis.pcolormesh(path.x_m*1e3, path.y_m*1e3,
                    abs(path.final_field)**2/1e4, shading="auto", cmap="viridis", rasterized=True)
                axis.set(xlabel="Camera x (mm)", ylabel="Camera y (mm)", aspect="equal")
                figure.colorbar(image, ax=axis, label="Intensity (W/cm^2)")
            apply_gabes_plot_style(figure)
            if view == "2D intensity":
                axis.grid(False)
            render_fig(figure)
        except BaseException:
            plt.close(figure)
            raise


def render_profiler(chain, *, temperature_c=100., cell_length_m=.05):
    """Render a standalone/chain-fed profiler even without a canvas probe."""
    _seed()
    st.markdown("###### Rb single-beam profiler")
    st.caption(MODEL_SCOPE)
    try:
        arguments, metadata = _controls(chain, temperature_c, cell_length_m)
    except (ValueError, TypeError, KeyError) as error:
        _remember_inputs()
        st.error(f"Cannot use this input: {error}. Choose Custom single beam or adjust the source.")
        return
    _remember_inputs()
    version = cache_version()
    signature = arguments + (version,)
    run = st.button("Run single-beam calculation", key=key("run"), type="primary")
    if run:
        progress = st.progress(0., text="Preparing single-beam response")
        try:
            result = _cached_cell(*arguments, version, _progress=lambda v, text: progress.progress(v, text=text))
            st.session_state[key("result")] = dict(signature=signature, cell=result, source=metadata,
                                                       numerical_fingerprint=version)
        except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as error:
            st.session_state.pop(key("result"), None)
            st.error(f"Calculation stopped: {error}. Check the inputs, enlarge the window or intensity ceiling, "
                     "or increase numerical resolution before running again.")
        finally:
            progress.empty()
    stored = st.session_state.get(key("result"))
    if stored is None:
        st.info("Set the cell and laser conditions, then run. The initial custom input is the Rb87 D1 blue reference.")
        return
    if stored["signature"] != signature:
        st.info("Inputs changed. Run single-beam calculation to update the profile.")
        return
    cell = stored["cell"]
    st.number_input("Camera distance after cell (mm)", min_value=0., max_value=5000., step=50.,
                    key=key("camera_mm"), help="Free space only. Changing this plane reuses the saved cell field.")
    distance = st.session_state[key("camera_mm")]*1e-3
    try:
        camera = _cached_camera(cell["input_key"], distance, version, _cell=cell)
    except (ValueError, FloatingPointError) as error:
        st.error(f"Camera plane unavailable: {error}. Reduce the distance or enlarge the window and rerun.")
        return
    if camera["checks"]["passed"]:
        st.success("Cell refinement passed; camera power and boundary checks passed.")
    elif not cell["checks"]["propagation"]["assessed"]:
        st.warning("Exploratory profile: propagation convergence has not been checked.")
    else:
        st.warning("Numerical refinement did not pass. Increase resolution and rerun before interpreting this profile.")
    path = camera["paths"]["full"]
    transmission = path.power_w[-1]/cell["incident_power_w"]
    radius = path.radius_m[-1]*1e3 if transmission >= model.DARK_TRANSMISSION else None
    metrics = [dict(label="Transmitted power", value=f"{path.power_w[-1]*1e3:.6g} mW"),
               dict(label="Transmission", value=f"{100*transmission:.6g} %"),
               dict(label="Second-moment radius", value="unavailable" if radius is None else f"{radius:.6g} mm")]
    st.markdown(strip_html(metrics, (), (), preserve_text=True), unsafe_allow_html=True)
    lens = cell["nonlinear_lens"]
    if lens.get("available"):
        sign = lens.get("sign_status", "numerical sign unresolved")
        st.caption(f"Cell-exit phase relative to absorption-only control: 1/f = {lens['lens_power_m_inverse']:+.6g} "
                   f"m\u207b\u00b9 | {sign} | phase fit RMS {lens['phase_rms_rad']:.3g} rad")
    else:
        st.caption("Equivalent lens unavailable: "+lens.get("status", "nonquadratic or dark wavefront"))
    view = st.radio("Profile view", ("Camera profile", "Beam radius", "2D intensity"),
                    horizontal=True, key=key("view"))
    _plots(cell, camera, view)
    st.caption("Camera plane is free space after the cell. FWM, collection lenses and downstream losses are not applied. "
               "Camera-plane radius/profile convergence is not independently assessed.")
    with st.expander("Conditions, convergence & export"):
        st.json(json_value(dict(source=metadata, conditions=cell["conditions"], checks=camera["checks"])))
        text, arrays = export_profile(cell, camera, metadata,
                                      calculation_fingerprint=stored["numerical_fingerprint"])
        left, right = st.columns(2)
        left.download_button("Download conditions & checks", text, file_name="sabes_single_beam.json",
                             mime="application/json", key=key("json"), width="stretch")
        right.download_button("Download field arrays", arrays, file_name="sabes_single_beam.npz",
                              mime="application/octet-stream", key=key("npz"), width="stretch")

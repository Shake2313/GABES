"""
Coefficients, and where each one came from.

SABES predicts lab quantities from lab settings, so it is only as good as the
numbers connecting them. The single most common way a digital twin goes wrong is
that a plausible-looking guess and a measured value are stored the same way and
become indistinguishable a month later. Every coefficient here therefore carries
a `provenance` tag, and the UI is expected to show it:

    fitted     solved from measured data
    datasheet  manufacturer specification for the exact part
    paper      stated in Sim et al., Sci. Rep. 15, 7727 (2025)
    lab        reported by the experimenter, not a formal measurement
    nominal    a placeholder not backed by a measurement
    hand       adjusted interactively during a session

The historical `verified` flag groups documented source types. Exact
applicability and independent validation are tracked separately in `evidence`.

Coefficients live in JSON so they can be replaced without touching code, which is
the whole point: when a real etalon scan or P-I curve arrives, only the file
changes.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict

import json

PROVENANCE_ORDER = ("fitted", "datasheet", "paper", "lab", "nominal", "hand")
VERIFIED_PROVENANCE = frozenset({"fitted", "datasheet", "paper"})

DEFAULT_CALIBRATION_PATH = Path(__file__).with_name("calibration_sim2025.json")
_EVIDENCE_FIELDS = (
    "evidence_class", "source", "applicability", "conditions", "uncertainty",
    "uncertainty_distribution", "review_date", "owner", "role", "fit",
)


@dataclass(frozen=True)
class Coefficient:
    name: str
    value: float
    unit: str = ""
    provenance: str = "nominal"
    note: str = ""
    evidence: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.provenance not in PROVENANCE_ORDER:
            raise ValueError(
                f"{self.name}: unknown provenance {self.provenance!r}; "
                f"expected one of {PROVENANCE_ORDER}")
        unknown = set(self.evidence) - set(_EVIDENCE_FIELDS)
        if unknown:
            raise ValueError(f"{self.name}: unknown evidence fields {sorted(unknown)}")
        fields = dict.fromkeys(_EVIDENCE_FIELDS)
        fields.update(evidence_class=self.provenance,
                      applicability={"status": "unknown", "scope": None})
        fields.update(deepcopy(self.evidence))
        if self.provenance == "fitted":
            fit = dict.fromkeys(("dataset", "objective", "companions",
                                 "covariance", "identifiability", "holdout"))
            fit.update(fields["fit"] or {})
            fields["fit"] = fit
        object.__setattr__(self, "evidence", fields)

    @property
    def verified(self):
        return self.provenance in VERIFIED_PROVENANCE

    @property
    def applicable(self):
        return self.evidence["applicability"]["status"] == "applicable"

    def as_dict(self):
        return {"value": self.value, "unit": self.unit,
                "provenance": self.provenance, "note": self.note,
                "evidence": deepcopy(self.evidence)}


class Calibration:
    """An immutable-by-convention bag of `Coefficient`s keyed by name."""

    def __init__(self, coefficients):
        self._items: Dict[str, Coefficient] = dict(coefficients)

    # ---- access ----
    def __contains__(self, name):
        return name in self._items

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items.values())

    def get(self, name) -> Coefficient:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"no calibration coefficient named {name!r}. "
                f"Add it to the calibration file rather than hard-coding it."
            ) from None

    def value(self, name, default=None):
        if default is not None and name not in self._items:
            return default
        return self.get(name).value

    def names(self):
        return tuple(sorted(self._items))

    # ---- mutation returns a new object, so a session can branch safely ----
    def with_value(self, name, value, provenance="hand", note=None, *, evidence=None):
        """Override one coefficient. Defaults to tagging the result `hand`."""
        current = self.get(name)
        updated = replace(
            current, value=float(value), provenance=provenance,
            note=current.note if note is None else note, evidence=evidence or {})
        items = dict(self._items)
        items[name] = updated
        return Calibration(items)

    def with_values(self, updates, provenance="hand"):
        """Override changed coefficients with one copy of the calibration map."""
        items = None
        for name, value in updates.items():
            current = self.get(name)
            value = float(value)
            if value == current.value:
                continue
            if items is None:
                items = dict(self._items)
            items[name] = replace(current, value=value, provenance=provenance, evidence={})
        return self if items is None else Calibration(items)

    def anchored_to(self, name, measured_value, note="", *, evidence=None):
        """Record an explicit fit anchor; retain unknown validation evidence."""
        return self.with_value(name, measured_value, provenance="fitted",
                               note=note or "anchored to a measured value",
                               evidence=evidence)

    # ---- reporting ----
    def unverified(self):
        """Unverified coefficients in declared provenance order."""
        return tuple(sorted(
            (c for c in self._items.values() if not c.verified),
            key=lambda c: (PROVENANCE_ORDER.index(c.provenance), c.name)))

    def evidence_counts(self):
        """Counts by source class and applicability, computed from the active profile."""
        return dict(Counter((c.evidence["evidence_class"],
                             c.evidence["applicability"]["status"]) for c in self))

    def table_rows(self):
        return tuple(
            (c.name, c.value, c.unit, c.provenance, c.note)
            for c in sorted(self._items.values(),
                            key=lambda c: (PROVENANCE_ORDER.index(c.provenance),
                                           c.name)))

    # ---- persistence ----
    def to_dict(self) -> Dict[str, Any]:
        return {name: self._items[name].as_dict() for name in sorted(self._items)}

    def output_evidence(self, *, gain_calibrated=False, recorded_transmission=False,
                        recorded_qe=False):
        """Document coefficient groups used by the current SABES readouts."""
        from .layout import parts

        def route_coefficients(*routes):
            return {parts.find_node(node).transmission_key for route in routes for node in route}
        coefficients = set(self._items)
        source = coefficients & {
            "loss_ecdl_to_ta", "ta_threshold_a", "ta_slope_w_per_a",
            "ta_saturation_w", "ta_ase_fraction", "ta_m2_out", "pump_fiber_efficiency",
            "seed_fiber_in_efficiency", "seed_fiber_out_efficiency", "waist_scale_factor",
            "pump_collimator_diameter_mm", "seed_collimator_diameter_mm", "cell_length_mm",
        }
        source |= {key for key in coefficients if key.startswith(("etalon_", "eom_"))}
        source |= coefficients & (route_coefficients(
            parts.ROUTE_TA_TO_SPLIT, parts.ROUTE_PUMP_TO_FIBER,
            parts.ROUTE_PUMP_DELIVERY, parts.ROUTE_SEED_TO_EOM,
            parts.ROUTE_SEED_FILTERS_TO_GT, parts.ROUTE_SEED_FILTERS_AFTER_NEL,
            parts.ROUTE_SEED_DELIVERY) | {"t_m_seed_out", "t_noise_eater"})
        post_cell = route_coefficients(
            parts.ROUTE_POST_CELL_PROBE, parts.ROUTE_POST_CELL_CONJUGATE)
        instrument = coefficients & {"bpd_nep_w_per_rthz", "bpd_cw_saturation_w",
                                     "pd_active_diameter_mm", "pd_max_power_density_w_per_cm2"}
        external = (set() if recorded_transmission
                    else coefficients & (post_cell | {"loss_post_cell_residual"}))
        if not recorded_qe:
            external |= coefficients & {"pd_responsivity_a_per_w"}
        outputs = {
            "gain": (source, ["reduced atomic source"] +
                     (["empirical gain correction and declared applicability"] if gain_calibrated else [])),
            "noise": (source | external,
                      ["selected effective noise model", "external optical loss and detector QE",
                       "microscopic covariance and matched SQL evidence unavailable"]),
            "detector_clearance": (source | external | instrument,
                                   ["stock electronics NEP used as an estimate for the modified detector",
                                    "equal-noise-band assumptions from the detector model"]),
        }
        return {
            key: {
                "coefficients": sorted(dependencies),
                "unknown_applicability_or_uncertainty": sorted(
                    name for name in dependencies if not self._items[name].applicable
                    or self._items[name].evidence["uncertainty"] is None),
                "assumptions": assumptions,
                "propagated_uncertainty": None,
            }
            for key, (dependencies, assumptions) in outputs.items()
        }

    def save(self, path):
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")

    @classmethod
    def from_dict(cls, payload):
        return cls({
            name: Coefficient(
                name=name,
                value=float(entry["value"]),
                unit=entry.get("unit", ""),
                provenance=entry.get("provenance", "nominal"),
                note=entry.get("note", ""),
                evidence=entry.get("evidence", {}))
            for name, entry in payload.items()
        })

    @classmethod
    def load(cls, path=None):
        path = Path(path) if path is not None else DEFAULT_CALIBRATION_PATH
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def default_calibration():
    """The Sim et al. setup as currently known. Reloaded from disk each call."""
    return Calibration.load()

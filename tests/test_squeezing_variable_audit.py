from gabes_lab.pipelines.squeezing.variable_audit import scan_variable_audit as audit


def test_paper_section_resolves_the_tracked_reference_table():
    result = audit.sec_paper_db()

    assert len(result["papers"]) == 10
    assert result["summary"]["cell_length_mm"]["n"] > 0


def test_gold_scan_cache_is_separate_for_each_velocity_measure(monkeypatch):
    calls = []

    def spectrum(*args, **kwargs):
        calls.append((kwargs["velocity_step"], kwargs["velocity_cutoff"], kwargs["velocity_measure"]))
        return {"probe_axis_GHz": 0.0}

    monkeypatch.setattr(audit, "_GOLD_SCANS", {})
    monkeypatch.setattr(audit, "RESPONSE_METHOD", audit.fwm.RESPONSE_POLE)
    monkeypatch.setattr(audit.fwm, "compute_spectrum", spectrum)
    converged = audit.scan(audit.GOLD)
    vstep, vcut = audit.HISTORICAL_VELOCITY_GRID
    historical = audit.scan(audit.GOLD, vstep=vstep, vcut=vcut)

    assert calls == [(1.0, 4.0, "exact"), (5.0, 3.0, "truncated")]
    assert converged is not historical
    assert audit.scan(audit.GOLD) is converged
    assert audit.scan(audit.GOLD, vstep=vstep, vcut=vcut) is historical


def test_grid_response_uses_converged_sampled_measure(monkeypatch):
    calls = []

    def spectrum(*args, **kwargs):
        calls.append((kwargs["velocity_step"], kwargs["velocity_cutoff"], kwargs["velocity_measure"]))
        return {"probe_axis_GHz": 0.0}

    monkeypatch.setattr(audit, "_GOLD_SCANS", {})
    monkeypatch.setattr(audit, "RESPONSE_METHOD", audit.fwm.RESPONSE_GRID)
    monkeypatch.setattr(audit.fwm, "compute_spectrum", spectrum)
    audit.scan(audit.GOLD)

    assert calls == [(1.0, 4.0, "truncated")]

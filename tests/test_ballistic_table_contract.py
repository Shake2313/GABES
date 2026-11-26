"""The Fast coefficient table follows upstream physics, independently of its readout."""

from gabes import ballistic_spectrum, species
from gabes._ballistic_reference_table import source_signature


def test_signature_tracks_physics_but_excludes_downstream_readout(monkeypatch):
    compute = source_signature.__wrapped__
    original = compute()
    monkeypatch.setattr(ballistic_spectrum, "_deplete", None)
    assert compute() == original

    old_width = species.self_broadening

    def changed_width(*args, **kwargs):
        return 1.05 * old_width(*args, **kwargs)

    monkeypatch.setattr(species, "self_broadening", changed_width)
    assert compute() != original

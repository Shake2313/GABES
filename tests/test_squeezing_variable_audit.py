from gabes_lab.pipelines.squeezing.variable_audit.scan_variable_audit import sec_paper_db


def test_paper_section_resolves_the_tracked_reference_table():
    result = sec_paper_db()

    assert len(result["papers"]) == 10
    assert result["summary"]["cell_length_mm"]["n"] > 0

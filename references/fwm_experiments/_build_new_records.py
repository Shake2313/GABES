"""Reproduce the additional-paper records from manually checked primary sources.

All source locations below refer to the PDF version identified by its checksum.
No fitting or graph digitization is performed by this script.
"""
import copy
import hashlib
import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent
DATE = "2026-10-23"


def q(name, value, unit, locator, *, kind="reported", basis="", reference=None,
      plane="not_specified", evidence="", notes="", uncertainty=None, **extra):
    row = dict(quantity=name, value=value, unit=unit, value_type=kind, basis=basis,
               reference=reference, plane=plane, source_id="main", locator=locator,
               evidence=evidence, notes=notes)
    if uncertainty is not None:
        row["uncertainty"] = uncertainty
    row.update(extra)
    return row


def missing(name, unit, locator, notes="Not specified for this operating point in the inspected source."):
    return q(name, None, unit, locator, kind="not_reported", notes=notes)


def rng(name, low, high, unit, locator, **kw):
    return q(name, None, unit, locator, kind="range", lower=low, upper=high, **kw)


def derived(name, value, unit, locator, formula, inputs, **kw):
    if name == "tpd_gabes":
        kw.setdefault("reference", "seed=pump-3035.732439MHz+delta (red Raman branch)")
    return q(name, value, unit, locator, kind="derived",
             derivation={"formula": formula, "inputs": inputs}, **kw)


def exp(id, label, conditions, results, kind="operating_point", notes=None, **extra):
    return dict(experiment_id=id, label=label, kind=kind,
                conditions=copy.deepcopy(conditions), results=results, notes=notes or [], **extra)


def paper(id, title, authors, year, journal, volume, pages, doi, source_url,
          experiments, *, arxiv="", path=None, version="published", group="",
          inclusion="primary", reason="85Rb D1 double-lambda FWM with experimental intensity-difference noise measurements.",
          composition="not_reported", notes=None):
    path = path or f"sources/{id}.pdf"
    p = (ROOT / path).resolve()
    doc = pymupdf.open(p)
    record = dict(schema_version=1, paper_id=id, title=title, authors=authors, year=year,
                  publication=dict(journal=journal, volume=volume, issue="", pages=pages,
                                   doi=doi, arxiv_id=arxiv, status="published"), legacy_column=None,
                  apparatus_group=group,
                  scope=dict(species="85Rb", line="D1", configuration="double_lambda",
                             measurement="intensity_difference", seed_mode="single_seed",
                             inclusion=inclusion, reason=reason, isotope_composition=composition),
                  sources=[dict(source_id="main", url=source_url, path=path, access="full_text",
                                version=version, sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                pages=len(doc), accessed_on=DATE)],
                  experiments=experiments, ai_notes=notes or [])
    atomic_used = False
    for experiment in record["experiments"]:
        formulas = [v.get("derivation", {}).get("formula", "") for v in experiment["conditions"] + experiment["results"]]
        for marker, name, value, unit, note in [
            ("3035.732439", "reference_ground_hyperfine_splitting", 3035.732439, "MHz", "Steck85RbD1 ground-state interval; conversion precision does not improve the paper's frequency setting."),
            ("361.58", "reference_excited_hyperfine_splitting", 361.58, "MHz", "Steck85RbD1 excited-state interval361.58(17)MHz; propagated origin uncertainty must be retained in quantitative fitting.")]:
            if any(marker in formula for formula in formulas):
                constant = q(name,value,unit,"PDF p.26, Fig.3",basis="atomic reference, not an apparatus setting",notes=note)
                constant["source_id"] = "atomic_reference"
                if name == "reference_excited_hyperfine_splitting":
                    constant["uncertainty"] = {"value":0.17,"unit":"MHz","kind":"reported_parenthetical","confidence":"not_specified"}
                experiment["conditions"].append(constant)
                for item in experiment["conditions"] + experiment["results"]:
                    if marker in item.get("derivation", {}).get("formula", ""):
                        item["derivation"]["inputs"].append(name)
                atomic_used = True
    if atomic_used:
        atomic_path = ROOT / "../Rubidium 85 D Line Data.pdf"
        atomic_doc = pymupdf.open(atomic_path)
        record["sources"].append(dict(source_id="atomic_reference",url="https://steck.us/alkalidata/rubidium85numbers.pdf",path="../Rubidium 85 D Line Data.pdf",access="full_text",version="local_Steck_reference",sha256=hashlib.sha256(atomic_path.read_bytes()).hexdigest(),pages=len(atomic_doc),accessed_on=DATE))
    (ROOT / "papers").mkdir(exist_ok=True)
    (ROOT / "papers" / f"{id}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return record


records = []

# Kim & Marino: the 87Rb transitions are targets for light generated in 85Rb.
k_setup = [q("cell_length",12,"mm","PDF p.3, §2"),
           q("detector_qe",95,"%","PDF p.3, §2",basis="photodiode quantum efficiency"),
           q("optical_transmission",96,"%","PDF p.5, §2.1",plane="after_cell_optics"),
           rng("cell_output_window_transmission",98,99,"%","PDF p.5, §2.1",plane="cell_exit_window"),
           q("detection_efficiency",90,"%","PDF p.5, §2.1",kind="approximate",basis="window, downstream optics and detector QE combined"),
           missing("seed_power","µW","PDF pp.3-7, §2")]
k_red = k_setup + [q("temperature",89,"°C","PDF p.4, §2.1"),
           q("pump_power",1000,"mW","PDF p.4, §2.1",plane="cell_input"),
           q("pump_beam_size",1900,"µm","PDF p.4, §2.1",basis="1/e² intensity diameter"),
           q("seed_beam_size",600,"µm","PDF p.4, §2.1",basis="1/e² intensity diameter"),
           derived("pump_radius",950,"µm","PDF p.4, §2.1","1/e² diameter / 2",["pump_beam_size"],basis="1/e² intensity radius"),
           derived("seed_radius",300,"µm","PDF p.4, §2.1","1/e² diameter / 2",["seed_beam_size"],basis="1/e² intensity radius"),
           q("crossing_angle",0.45,"deg","PDF p.4, §2.1"),
           q("tpd_reported",16,"MHz","PDF pp.4,6, §2.1",basis="pump-probe separation minus ground hyperfine splitting"),
           derived("tpd_gabes",-16,"MHz","PDF pp.4,6, §2.1","-tpd_reported",["tpd_reported"],notes="Sign also checked against the reported 6.104-GHz probe-conjugate beat; rounded paper detuning retained."),
           q("analysis_frequency",750,"kHz","PDF pp.4-5, Fig.3"),
           q("resolution_bandwidth",30,"kHz","PDF p.5, Fig.3 caption"),
           q("video_bandwidth",100,"Hz","PDF p.5, Fig.3 caption")]
k_probe = k_red + [q("opd_reported",-687,"MHz","PDF p.4, §2.1",reference="85Rb F=2→F′=2",basis="negative means red"),
           derived("opd_gabes",-1.04858,"GHz","PDF p.4, §2.1; Steck PDF p.26 Fig.3","(-687 - 361.58) / 1000",["opd_reported"],reference="85Rb F=2→F′=3",notes="Uses361.58(17)-MHz excited-state hyperfine splitting; this is a basis conversion, not another reported detuning.")]
k_conj = k_red + [q("opd_reported",-771,"MHz","PDF p.4, §2.1",reference="85Rb F=2→F′=2",basis="negative means red"),
           derived("opd_gabes",-1.13258,"GHz","PDF p.4, §2.1; Steck PDF p.26 Fig.3","(-771 - 361.58) / 1000",["opd_reported"],reference="85Rb F=2→F′=3")]
k_double = k_setup + [q("temperature",91,"°C","PDF p.7, §2.2"),q("pump_power",1000,"mW","PDF p.7, §2.2",plane="cell_input"),q("pump_beam_size",1900,"µm","PDF p.7, §2.2",basis="1/e² intensity diameter"),q("seed_beam_size",600,"µm","PDF p.7, §2.2",basis="1/e² intensity diameter"),derived("pump_radius",950,"µm","PDF p.7, §2.2","diameter / 2",["pump_beam_size"],basis="1/e² intensity radius"),derived("seed_radius",300,"µm","PDF p.7, §2.2","diameter / 2",["seed_beam_size"],basis="1/e² intensity radius"),q("crossing_angle",0.5,"deg","PDF p.7, §2.2"),q("tpd_reported",-26,"MHz","PDF p.7, §2.2",basis="pump-probe separation minus ground hyperfine splitting"),derived("tpd_gabes",26,"MHz","PDF p.7, §2.2","-tpd_reported",["tpd_reported"]),q("opd_reported",-687,"MHz","PDF pp.6-7, §2.2",reference="85Rb F=2→F′=2",basis="negative means red"),derived("opd_gabes",-1.04858,"GHz","PDF pp.6-7, §2.2; Steck PDF p.26 Fig.3","(-687 - 361.58) / 1000",["opd_reported"],reference="85Rb F=2→F′=3"),q("analysis_frequency",750,"kHz","PDF p.7, Fig.5"),q("resolution_bandwidth",30,"kHz","PDF p.7, Fig.5 caption"),q("video_bandwidth",100,"Hz","PDF p.7, Fig.5 caption")]
records.append(paper("kim_2018_oe","Generation of 87Rb resonant bright two-mode squeezed light with four-wave mixing",["Saesun Kim","Alberto M. Marino"],2018,"Optics Express","26","33366-33375","10.1364/OE.26.033366","https://doi.org/10.1364/OE.26.033366",[
    exp("probe_resonant","Probe resonant with 87Rb F=2→F′=2",k_probe,[q("probe_gain",5.6,"1","PDF p.5, §2.1",basis="measured FWM gain; precise input/output planes not separately stated"),q("squeezing_measured",-5.4,"dB","PDF p.4, §2.1"),q("squeezing_electronic_noise_subtracted",-6.3,"dB","PDF pp.4,6, Fig.4(a)"),q("squeezing_loss_corrected",-8.3,"dB","PDF p.5, §2.1",basis="source noise inferred using approximately90% detection efficiency")]),
    exp("conjugate_resonant","Conjugate resonant with 87Rb F=1→F′=1",k_conj,[q("probe_gain",3.8,"1","PDF p.5, §2.1",basis="measured FWM gain; precise input/output planes not separately stated"),q("squeezing_measured",-5.0,"dB","PDF p.4, §2.1"),q("squeezing_electronic_noise_subtracted",-6.2,"dB","PDF pp.4,6, Fig.4(b)"),q("squeezing_loss_corrected",-8.1,"dB","PDF p.5, §2.1",basis="source noise inferred using approximately90% detection efficiency")]),
    exp("doubly_resonant","Probe and conjugate both on 87Rb resonance",k_double,[q("probe_gain",1.9,"1","PDF p.7, §2.2",basis="measured FWM gain"),q("squeezing_measured",-3.5,"dB","PDF p.7, §2.2"),q("squeezing_electronic_noise_subtracted",-3.9,"dB","PDF p.7, §2.2"),q("squeezing_loss_corrected",-4.7,"dB","PDF p.7, §2.2")]),
    exp("off_resonance_reference","Typical blue-detuned off-resonance reference from earlier work",[q("opd_reported",800,"MHz","PDF p.3, Fig.1(a), §2",kind="approximate",basis="blue-detuned off-resonance configuration",reference="85Rb F=2→F′=3 schematic reference"),q("pump_power",400,"mW","PDF p.3, §2",kind="approximate"),q("pump_beam_size",1300,"µm","PDF p.3, §2",basis="1/e² intensity diameter"),q("seed_beam_size",700,"µm","PDF p.3, §2",basis="1/e² intensity diameter"),derived("pump_radius",650,"µm","PDF p.3, §2","diameter / 2",["pump_beam_size"],basis="1/e² intensity radius"),derived("seed_radius",350,"µm","PDF p.3, §2","diameter / 2",["seed_beam_size"],basis="1/e² intensity radius"),q("crossing_angle",0.3,"deg","PDF p.3, §2"),q("tpd_reported",4,"MHz","PDF p.3, §2",basis="paper spacing convention"),missing("temperature","°C","PDF p.3, §2"),missing("seed_power","µW","PDF p.3, §2")],[q("squeezing_measured",-9,"dB","PDF p.3, Fig.1(a), §2",basis="previously demonstrated reference, not a new independent measurement in this paper")],kind="literature_summary",notes=["The −9-dB reference is the blue-detuned configuration; do not assign it to the red-detuned 87Rb-resonant operating points."])
],path="../oe-26-25-33366.pdf",group="oklahoma_marino_rb85",notes=["⟦AI⟧ Original-title isotope87 labels the target resonances, not the FWM-medium isotope. Source is85Rb.","⟦AI⟧ Off-resonance summary is excluded from independent measurement counts. No seed power was borrowed from the cited McCormick experiment."]))

# Jasperse: a measured sweep, not a table of independent exact nodes.
j = [q("pump_power",400,"mW","PDF p.8, §5",plane="cell_input"),q("seed_power",100,"µW","PDF p.8, §5",plane="cell_input"),q("pump_beam_size",630,"µm","PDF p.8, §5",basis="1/e² beam waist"),q("seed_beam_size",375,"µm","PDF p.8, §5",basis="1/e² beam waist"),derived("pump_radius",630,"µm","PDF p.8, §5","reported1/e² waist interpreted as radius",["pump_beam_size"],basis="1/e² intensity radius"),derived("seed_radius",375,"µm","PDF p.8, §5","reported1/e² waist interpreted as radius",["seed_beam_size"],basis="1/e² intensity radius"),q("crossing_angle",0.3,"deg","PDF p.8, §5"),q("cell_length",7,"mm","PDF p.8, §5",basis="internal cell length"),q("temperature",130,"°C","PDF p.8, §5"),q("pump_probe_spacing",3040,"MHz","PDF p.8, §5",basis="probe below pump"),derived("tpd_gabes",-4.267561,"MHz","PDF p.8, §5","3035.732439 - pump_probe_spacing",["pump_probe_spacing"],notes="Precision reflects the conversion constant, not the rounded3040-MHz experimental setting."),q("detector_qe",95,"%","PDF p.8, §5",basis="net photodiode efficiency"),q("detection_efficiency",85,"%","PDF p.8, §5",uncertainty={"value":1,"unit":"percentage_point","kind":"reported","confidence":"not_specified"}),q("analysis_frequency",1000,"kHz","PDF p.8, §5"),q("resolution_bandwidth",30,"kHz","PDF p.8, §5"),missing("opd_gabes","GHz","PDF p.9, Fig.5 caption","Pump frequency is referenced to the centre of the Doppler-broadened F=2→5P1/2 manifold, not an identified F′=3 line.")]
records.append(paper("jasperse_2011_oe","Relative intensity squeezing by four-wave mixing with loss: an analytic model and experimental diagnostic",["M. Jasperse","L. D. Turner","R. E. Scholten"],2011,"Optics Express","19","3765-3774","10.1364/OE.19.003765","https://arxiv.org/pdf/1012.3482",[
    exp("pump_detuning_sweep","Fig.5 measured two-arm gain and squeezing sweep",j,[missing("probe_gain","1","PDF p.9, Fig.5(A)","Measured curve exists; exact numerical nodes are not tabulated and were not digitized."),missing("conjugate_gain","1","PDF p.9, Fig.5(A)","Measured curve exists; exact numerical nodes are not tabulated and were not digitized."),missing("squeezing_measured","dB","PDF p.9, Fig.5(C)","Measured curve exists; exact minimum not stated numerically in the text."),q("squeezing_pump_background_subtracted",-4.1,"dB","PDF p.9, §5",basis="author estimate after removal of cross-detected pump background; not a raw measured best")],kind="sweep",notes=["Intrinsic gainG and probe transmissionTa in Fig.5(B) are fitted distributed-medium parameters, not directly measured arm gains.","Detuning-axis origin: centre of85Rb5S1/2(F=2)→5P1/2 manifold. Curve digitization required before pointwise fitting."])
],arxiv="1012.3482v2",version="arxiv_v2",group="melbourne_monash_scholten",composition="pure85",notes=["⟦AI⟧ Experimental gain curves can constrain a future correction after documented digitization; no plot-axis values have been promoted to measurements."]))

# Glorieux: retain Fig.7 detector configuration separately from earlier sweeps.
gbase=[q("cell_length",12.5,"mm","PDF p.2, §2"),q("pump_beam_size",650,"µm","PDF p.2, §2",basis="radius at1/e²"),q("seed_beam_size",350,"µm","PDF p.2, §2",basis="radius at1/e²"),derived("pump_radius",650,"µm","PDF p.2, §2","identity: explicit radius",["pump_beam_size"],basis="1/e² intensity radius"),derived("seed_radius",350,"µm","PDF p.2, §2","identity: explicit radius",["seed_beam_size"],basis="1/e² intensity radius"),q("crossing_angle",1,"mrad","PDF p.2, §2",kind="approximate"),derived("crossing_angle_deg",0.0572957795,"deg","PDF p.2, §2","1 mrad *180/(1000*pi)",["crossing_angle"],notes="Underlying angle is approximate; extra decimals are conversion only."),q("resolution_bandwidth",100,"kHz","PDF p.2, §2"),q("video_bandwidth",10,"Hz","PDF p.2, §2"),missing("seed_power","µW","PDF pp.2-6","Probe availability0–300µW is an apparatus envelope; the optimum-point input power is not specified."),missing("opd_gabes","GHz","PDF p.4, §3.2","§3.2 labels the detuning from F=3→5P1/2 while the unresolved-excited-state scheme does not establish a unique F′=3 origin; do not guess an offset.")]
gopt=gbase+[q("temperature",118,"°C","PDF p.6, Fig.7 caption"),q("pump_power",1200,"mW","PDF p.6, Fig.7 caption"),q("opd_reported",750,"MHz","PDF p.6, Fig.7 caption",reference="paper Δ; see conflicting F=3→5P1/2 description in§3.2"),q("tpd_reported",6,"MHz","PDF p.6, Fig.7 caption",basis="paper δ sign; formal frequency equation not supplied"),q("analysis_frequency",1000,"kHz","PDF p.6, Fig.7 caption",kind="approximate"),q("cell_output_window_loss",1.5,"%","PDF p.6, §3.5",uncertainty={"value":0.5,"unit":"percentage_point","kind":"reported","confidence":"not_specified"}),q("downstream_optical_loss",3,"%","PDF p.6, §3.5",basis="PBS, iris, mirrors combined",uncertainty={"value":1,"unit":"percentage_point","kind":"estimated","confidence":"not_specified"}),missing("detector_qe","%","PDF p.6, §3.5","The paper explicitly says photodiode QE cannot be accurately determined because detector gain is unknown.")]
gexps=[exp("best_improved_detector","Fig.7 optimized IDS with opened S3883 detectors",gopt,[q("squeezing_electronic_noise_subtracted",-9.2,"dB","PDF p.6, Fig.7",uncertainty={"value":0.5,"unit":"dB","kind":"reported","confidence":"not_specified"}),q("squeezing_loss_corrected",-11.0,"dB","PDF p.7, §3.5",basis="corrected only for propagation/cell losses, assuming perfect photodiode QE",uncertainty={"value":0.7,"unit":"dB","kind":"reported","confidence":"not_specified"}),missing("probe_gain","1","PDF p.6, Fig.7","No numeric gain given for the1200-mW best-noise setting; Fig.6 is a different sweep."),missing("squeezing_measured","dB","PDF p.6, Fig.7","The shown−9.2dB has electronic noise removed; an uncorrected number is not provided.")],notes=["Improved opened HamamatsuS3883 detectors replace the earlier commercial balanced receiver."])]
for eid,label,pump,temp,opd,tpd,loc,scan in [("temperature_sweep","Fig.3 temperature sweep",500,None,800,10,"PDF p.4, Fig.3 caption","temperature"),("opd_sweep","Fig.4 one-photon-detuning sweep",800,114,None,10,"PDF p.5, Fig.4 caption","opd_reported"),("tpd_sweep","Fig.5 two-photon-detuning sweep",800,114,800,None,"PDF p.5, Fig.5 caption","tpd_reported"),("pump_sweep","Fig.6 pump-power and probe-gain sweep",None,114,800,10,"PDF p.6, Fig.6 caption","pump_power")]:
    cc=copy.deepcopy(gbase)+[q("analysis_frequency",1500,"kHz",loc)]
    for n,val,u in [("pump_power",pump,"mW"),("temperature",temp,"°C"),("opd_reported",opd,"MHz"),("tpd_reported",tpd,"MHz")]:
        if val is not None:cc.append(q(n,val,u,loc,reference="paper Δ with unresolved excited state" if n=="opd_reported" else None))
    if scan=="temperature":cc.append(rng("temperature",110,126,"°C","PDF p.3, §3.1",basis="reported scan range"))
    gexps.append(exp(eid,label,cc,[missing("probe_gain","1",loc,"Curve only; no exact nodes digitized."),missing("squeezing_measured","dB",loc,"Curve only; no exact nodes digitized.")],kind="sweep",notes=[f"Sweep variable:{scan}; fixed controls retained at this figure's values. Do not borrow the Fig.7 optimum."]))
records.append(paper("glorieux_2010_spie","Strong quantum correlations in four wave mixing in 85Rb vapor",["Quentin Glorieux","Luca Guidoni","Samuel Guibal","Jean-Pierre Likforman","Thomas Coudreau"],2010,"Proceedings of SPIE","7727","772703","10.1117/12.854953","https://arxiv.org/pdf/1004.3950",gexps,arxiv="1004.3950v1",version="arxiv_v1",group="paris_mpq_glorieux",composition="pure85",notes=["⟦AI⟧ Article has an unresolved/possibly inconsistent OPD reference. Numeric original Δ values are preserved; no canonical F′=3 detuning guessed.","⟦AI⟧−9.2dB is electronic-noise-subtracted;−11dB is an authors' loss correction with a perfect-QE assumption."]))

# Qin2012: setup maximum power differs from the measurement's280mW.
qb=[q("temperature",121,"°C","PDF p.2, setup"),q("cell_length",12.5,"mm","PDF p.2, setup"),q("opd_reported",0.8,"GHz","PDF p.1, setup",kind="approximate",reference="85Rb F=2→5P1/2 unresolvedF′",basis="blue detuning"),missing("opd_gabes","GHz","PDF p.1, setup","Excited hyperfine reference is unspecified."),q("pump_probe_spacing",3040,"MHz","PDF pp.1-2, setup",basis="probe red of pump; nominal rounded offset"),derived("tpd_gabes",-4.267561,"MHz","PDF pp.1-2, setup","3035.732439 - pump_probe_spacing",["pump_probe_spacing"],notes="Rounded3.04GHz offset; not a MHz-accurate optimized detuning."),q("pump_beam_size",530,"µm","PDF p.2, setup",basis="1/e² radius"),q("seed_beam_size",330,"µm","PDF p.2, setup",basis="1/e² radius"),derived("pump_radius",530,"µm","PDF p.2, setup","identity: explicit radius",["pump_beam_size"],basis="1/e² intensity radius"),derived("seed_radius",330,"µm","PDF p.2, setup","identity: explicit radius",["seed_beam_size"],basis="1/e² intensity radius"),q("crossing_angle",0.3,"deg","PDF p.2, setup"),q("detector_qe",96,"%","PDF p.2, detection"),q("resolution_bandwidth",30,"kHz","PDF p.2, detection"),q("video_bandwidth",300,"Hz","PDF p.2, detection")]
qbest=qb+[q("pump_power",280,"mW","PDF p.2, operating-point paragraph",plane="cell_input"),q("seed_power",15,"µW","PDF p.2, operating-point paragraph",plane="cell_input"),q("analysis_frequency",850,"kHz","PDF p.2, Fig.2 discussion")]
qpowers=[q("output_probe_power",123,"µW","PDF p.2, operating-point paragraph",kind="approximate",plane="output_measurement_not_fully_specified"),q("output_conjugate_power",111,"µW","PDF p.2, operating-point paragraph",kind="approximate",plane="output_measurement_not_fully_specified"),derived("probe_gain",8.2,"1","PDF p.2, operating-point paragraph","output_probe_power / seed_power",["output_probe_power","seed_power"],basis="ratio of reported approximate optical powers; planes not fully specified"),derived("conjugate_gain",7.4,"1","PDF p.2, operating-point paragraph","output_conjugate_power / seed_power",["output_conjugate_power","seed_power"],basis="ratio of reported approximate optical powers; planes not fully specified")]
qlow=[v for v in copy.deepcopy(qb) if v['quantity'] not in ('pump_probe_spacing','tpd_gabes','resolution_bandwidth','video_bandwidth')]+[q("tpd_reported",14,"MHz","PDF p.2, low-frequency paragraph",basis="paperδ; formal sign not specified"),q("resolution_bandwidth",200,"Hz","PDF p.2, low-frequency paragraph"),missing("pump_power","mW","PDF pp.2-3, low-frequency measurement","The low-frequency paragraph does not restate pump power;280mW is not silently copied."),missing("seed_power","µW","PDF pp.2-3, low-frequency measurement")]
records.append(paper("qin_2012_ol","Compact diode-laser-pumped quantum light source based on four-wave mixing in hot rubidium vapor",["Zhongzhong Qin","Jietai Jing","Jun Zhou","Cunjin Liu","Raphael C. Pooser","Zhifan Zhou","Weiping Zhang"],2012,"Optics Letters","37","3141-3143","10.1364/OL.37.003141","https://zhifanzhou.com/files/Compact%20diode-laser-pumped%20quantum%20light%20source%20based.pdf",[
exp("unbalanced_best","Fig.2(c) unattenuated twin-beam output",qbest,qpowers+[q("squeezing_measured",-7,"dB","PDF p.2, Fig.2(c) discussion"),q("bandwidth",6,"MHz","PDF p.2, Fig.2(c) discussion",basis="squeezing frequency range")]),
exp("balanced_output","Fig.2(b) probe attenuation to balance detector powers",qbest,[q("squeezing_measured",-5.7,"dB","PDF p.2, Fig.2(b) discussion"),q("bandwidth",6.7,"MHz","PDF p.2, Fig.2(b) discussion")],kind="comparison",notes=["Probe attenuated after generation; source powers/gains must not be replaced by the balanced detector powers."]),
exp("noise_slope","Fig.3 seed-power sweep and noise slope",[v for v in copy.deepcopy(qbest) if v['quantity'] not in ('seed_power','analysis_frequency')]+[q("analysis_frequency",635,"kHz","PDF p.2, Fig.3")],[q("noise_ratio",0.213,"1","PDF p.2, Fig.3",basis="FWM/SQL fitted slope ratio"),q("squeezing_slope",-6.7,"dB","PDF p.2, Fig.3")],kind="sweep"),
exp("low_frequency","Fig.4 lower Fourier-frequency configuration",qlow,[q("output_probe_power",70,"µW","PDF p.2, low-frequency paragraph"),q("output_conjugate_power",64,"µW","PDF p.2, low-frequency paragraph"),q("squeezing_measured",-5,"dB","PDF p.3, low-frequency discussion",kind="approximate",basis="nearly flat noise above23kHz excluding technical peaks"),q("lowest_squeezed_frequency",8,"kHz","PDF p.3, low-frequency discussion"),q("flat_squeezing_frequency_start",23,"kHz","PDF p.3, low-frequency discussion")])
],group="ecnu_jing_diode_rb85",composition="pure85",notes=["⟦AI⟧400mW is available tapered-amplifier power, not the280mW used for the quoted123/111µW operating point. Low-frequencyδ14MHz is a separate configuration."]))

# Pooser: keep only the requested isotope/line as calibration observations.
pb=[q("cell_length",12,"mm","PDF p.4, §2"),q("temperature",110,"°C","PDF p.4, §2",kind="approximate"),q("crossing_angle",1,"deg","PDF p.4, §2",kind="upper_bound",basis="theta<1deg"),q("pump_power",400,"mW","PDF p.6, §3.1"),q("pump_beam_size",550,"µm","PDF p.6, §3.1",basis="1/e² intensity radius"),derived("pump_radius",550,"µm","PDF p.6, §3.1","identity: explicit radius",["pump_beam_size"],basis="1/e² intensity radius"),q("analysis_frequency",1000,"kHz","PDF p.4, §2"),q("detector_qe",94,"%","PDF p.4, §2",kind="approximate",basis="nominal photodiode efficiency"),q("detection_efficiency",90,"%","PDF p.4, §2",uncertainty={"value":3,"unit":"percentage_point","kind":"combined_statistical_and_systematic_1sigma","confidence":"1sigma"}),q("opd_reported",0.8,"GHz","PDF p.5, §3.1",reference="Doppler broadened peak from85RbF=2 ground state",basis="blue detuning"),missing("opd_gabes","GHz","PDF p.5, §3.1","Excited hyperfine structure unresolved; peak origin is not identified with F′=3."),missing("tpd_reported","MHz","PDF pp.4-5, §2","Two-photon detuning optimized then held fixed; numeric optimum not given."),missing("seed_power","µW","PDF pp.3-6"),missing("seed_radius","µm","PDF pp.3-6")]
records.append(paper("pooser_2009_oe","Quantum correlated light beams from non-degenerate four-wave mixing in an atomic vapor: the D1 and D2 lines of 85Rb and 87Rb",["R. C. Pooser","A. M. Marino","V. Boyer","K. M. Jones","P. D. Lett"],2009,"Optics Express","17","16722-16730","10.1364/OE.17.016722","https://pdfs.semanticscholar.org/bdcd/59d7f5e71388ea71afdea84f2d297ff85291.pdf",[
exp("rb85_d1_best","85Rb D1 optimum along Fig.2 detuning cut",pb,[q("probe_gain",5,"1","PDF p.6, §3.3 comparison",kind="approximate",basis="measured probe output / probe input; explicitly distinguished from inferredG"),q("squeezing_background_subtracted",-8,"dB","PDF p.5, §3.1; subtraction stated p.4 §2",uncertainty={"value":0.2,"unit":"dB","kind":"combined_statistical_and_systematic_1sigma","confidence":"1sigma"},basis="electronics and scattered-pump noise subtracted")],notes=["Other85RbD2/87RbD1/87RbD2 measurements are outside the requested calibration system and not imported as85RbD1 observations.","TheG andT curves in Fig.3 are model-inferred distributed gain/absorption parameters; do not substitute them for measured probe/conjugate gains."]),
exp("detuning_cut","Fig.2 85RbD1 noise vs pump detuning",[v for v in copy.deepcopy(pb) if v['quantity']!='opd_reported'],[missing("probe_gain","1","PDF p.6, Fig.3","Underlying measured arm powers were used to inferG andT, but exact arm-gain nodes are not tabulated."),missing("squeezing_background_subtracted","dB","PDF p.6, Fig.2","Curve retained as a source for future documented digitization.")],kind="sweep")
],group="nist_lett_rb85",notes=["⟦AI⟧The D1/85Rb subset is selected explicitly; this mixed-isotope/mixed-line article is not treated as four equivalent training records."]))

# Qin2014 single-cell controls retained separately from cascaded/noisy-seed records.
cb=[q("cell_length",12.5,"mm","PDF p.2, setup"),q("pump_power",210,"mW","PDF p.2, setup",kind="approximate"),q("pump_beam_size",550,"µm","PDF p.2, setup",kind="approximate",basis="beam waist;1/e² convention not explicitly restated"),q("seed_beam_size",280,"µm","PDF p.2, setup",basis="beam waist;1/e² convention not explicitly restated"),q("crossing_angle",6,"mrad","PDF p.2, setup"),derived("crossing_angle_deg",0.3437746771,"deg","PDF p.2, setup","6mrad *180/(1000*pi)",["crossing_angle"]),q("opd_reported",0.8,"GHz","PDF p.2, setup",kind="approximate",reference="85Rb F=2→5P1/2 unresolvedF′",basis="blue detuning"),missing("opd_gabes","GHz","PDF p.2, setup","F′ reference is not resolved."),q("pump_probe_spacing",3040,"MHz","PDF p.2, setup",basis="nominal probe red offset"),derived("tpd_gabes",-4.267561,"MHz","PDF p.2, setup","3035.732439 - pump_probe_spacing",["pump_probe_spacing"],notes="Nominal3.04GHz offset, not a precisely reported Raman detuning."),q("detector_qe",96,"%","PDF p.3, detection"),q("resolution_bandwidth",30,"kHz","PDF p.3, Fig.3 discussion"),q("video_bandwidth",300,"Hz","PDF p.3, Fig.3 discussion")]
c1=cb+[q("temperature",110,"°C","PDF p.2, setup",kind="approximate"),q("seed_power",20,"µW","PDF p.2, setup",kind="approximate")]
c2=cb+[q("temperature",108,"°C","PDF p.2, setup",kind="approximate"),missing("seed_power","µW","PDF pp.2-4","Single-cell Cell2 control seed power is not restated; the cascaded seed is amplified/noisy and is a separate configuration.")]
records.append(paper("qin_2014_prl","Experimental Generation of Multiple Quantum Correlated Beams from Hot Rubidium Vapor",["Zhongzhong Qin","Leiming Cao","Hailong Wang","A. M. Marino","Weiping Zhang","Jietai Jing"],2014,"Physical Review Letters","113","023602","10.1103/PhysRevLett.113.023602","https://arxiv.org/pdf/1407.3863",[
exp("cell1_single","Cell1 single-FWM twin-beam control",c1,[q("probe_gain",2.9,"1","PDF p.2, setup",kind="approximate",basis="first-cell gainG1"),q("output_probe_power",58,"µW","PDF p.2, setup",kind="approximate"),q("output_conjugate_power",41,"µW","PDF p.2, setup",kind="approximate"),q("squeezing_measured",-5.4,"dB","PDF p.4, Fig.3(b) discussion",kind="approximate",uncertainty={"value":0.4,"unit":"dB","kind":"reported","confidence":"not_specified"}),q("squeezing_slope",-5.5,"dB","PDF p.4, Fig.4",basis="electronics/background-subtracted fitted slope at1MHz",uncertainty={"value":0.1,"unit":"dB","kind":"reported","confidence":"not_specified"}),q("noise_ratio",0.282,"1","PDF p.4, Fig.4",basis="background-subtracted slope ratio",uncertainty={"value":0.003,"unit":"1","kind":"reported","confidence":"not_specified"})]),
exp("cell2_single","Cell2 single-FWM twin-beam control",c2,[q("squeezing_measured",-4.4,"dB","PDF p.4, Fig.3(c) discussion",kind="approximate",uncertainty={"value":0.4,"unit":"dB","kind":"reported","confidence":"not_specified"}),q("squeezing_slope",-4.5,"dB","PDF p.4, Fig.4",basis="electronics/background-subtracted fitted slope at1MHz",uncertainty={"value":0.1,"unit":"dB","kind":"reported","confidence":"not_specified"}),q("noise_ratio",0.356,"1","PDF p.4, Fig.4",basis="background-subtracted slope ratio",uncertainty={"value":0.003,"unit":"1","kind":"reported","confidence":"not_specified"}),missing("probe_gain","1","PDF pp.2-4","G2≈2.1 is reported for the second stage seeded byPr1; not automatically assigned to its independent coherent-seed control.")]),
exp("cascaded_cell2","Second stage seeded by amplified first-stage probe",cb+[q("temperature",108,"°C","PDF p.2, setup",kind="approximate"),q("interstage_optical_transmission",93,"%","PDF p.3, setup"),derived("seed_power",53.94,"µW","PDF pp.2-3, setup","58µW *0.93",["cell1_single.output_probe_power","interstage_optical_transmission"],basis="derived incident second-stage probe",notes="Seed is not a coherent-state input; first-stage excess noise/correlations matter.")],[q("probe_gain",2.1,"1","PDF p.2, setup",kind="approximate",basis="second-stageG2 relative to transmittedPr1"),q("output_probe_power",115,"µW","PDF p.3, setup",kind="approximate"),q("output_conjugate_power",71,"µW","PDF p.3, setup",kind="approximate")],kind="comparison",notes=["Do not use as a single-stage coherent-seed noise observation; seed has first-stage correlations."]),
exp("three_beam_cascade","Two cascaded cells: three-beam intensity difference",[],[q("three_beam_squeezing_measured",-6.5,"dB","PDF p.4, Fig.3(a) discussion",uncertainty={"value":0.4,"unit":"dB","kind":"reported","confidence":"not_specified"}),q("three_beam_squeezing_slope",-7,"dB","PDF p.4, Fig.4",basis="electronic/background-subtracted three-beam slope",uncertainty={"value":0.1,"unit":"dB","kind":"reported","confidence":"not_specified"})],kind="comparison",notes=["Three-beam observablePr2−C2−C1, not the single-cell two-arm IDS target."])
],arxiv="1407.3863v1",version="arxiv_v1",group="ecnu_jing_diode_rb85",notes=["⟦AI⟧Single-cell controls and cascaded amplified-seed measurements are distinct records. The headline−7dB concerns three beams and cannot serve as a single-cell IDS target."]))

# Turnbull contains direct experimental gain maps, but not a new IDS measurement.
tb=[q("temperature",110,"°C","PDF p.5, §III",kind="approximate"),q("cell_length",12,"mm","PDF p.5, §III"),q("pump_power",750,"mW","PDF p.5, §III"),q("pump_beam_size",900,"µm","PDF p.5, §III",basis="beam waist; intensity-radius definition not explicitly specified"),q("seed_beam_size",400,"µm","PDF p.5, §III",basis="beam waist; intensity-radius definition not explicitly specified"),rng("seed_power",10,20,"µW","PDF p.5, §III"),rng("crossing_angle",0,1,"deg","PDF p.5, §III",basis="experimental angular sweep"),q("tpd_scan_span",60,"MHz","PDF p.5, §III",kind="approximate",basis="typical two-photon scan width"),q("opd_reported",140,"Gamma","PDF p.6, §III",reference="Δ1 in unresolved-excited-state model;γ=2π×6MHz",basis="spectroscopy-calibrated pump detuning"),derived("opd_frequency",840,"MHz","PDF pp.2,6, §II/III","140 *6MHz",["opd_reported"],reference="paperΔ1 unresolved excited manifold"),missing("opd_gabes","GHz","PDF pp.2,6","Unresolved excited-state reference is not a uniquely specified85RbF′=3 transition.")]
records.append(paper("turnbull_2013_pra","Role of the phase-matching condition in nondegenerate four-wave mixing in hot vapors for the generation of squeezed states of light",["M. T. Turnbull","P. G. Petrov","C. S. Embrey","A. M. Marino","V. Boyer"],2013,"Physical Review A","88","033845","10.1103/PhysRevA.88.033845","https://arxiv.org/pdf/1303.7187",[
exp("angle_tpd_gain_map","Fig.7 measured probe/conjugate gain vs angle and detuning",tb,[missing("probe_gain","1","PDF p.6, Fig.7(a)","Measured color-map nodes not tabulated; digitization needed."),missing("conjugate_gain","1","PDF p.6, Fig.7(b)","Measured color-map nodes not tabulated; digitization needed.")],kind="sweep",notes=["Definitions are explicit:gp=Pprobe,out/Pseed,in;gc=Pconjugate,out/Pseed,in.","FittedΩ=60γ,N=2.8e12cm−3,γc=0.2γ,epsilon=6.5e−6 are model-inferred parameters, not independent apparatus measurements.","Theoretical gains350/1100 in Fig.7 caption are not experimental extrema."])
],arxiv="1303.7187v1",version="arxiv_v1",group="birmingham_boyer_rb85",inclusion="context_only",reason="85RbD1 double-lambda direct two-arm gain experiment useful for futuregain correction; this paper's new experiment does not report an IDS measurement.",notes=["⟦AI⟧Retained as gain-only context. Do not count theoretical noise predictions as measured squeezing. Published title uses'nondegenerate'; arXiv title uses'non-degenerate'."]))

screen = {
    "searched_on": DATE,
    "origin": "ai_research_for_user_requested_database",
    "queries": ["85Rb four wave mixing intensity difference squeezing Jasperse2011 near resonance", "85Rb four wave mixing Qin2012/2014 intensity difference squeezing", "Turnbull2013 phase matching hot vapors squeezed states", "Pooser2009 D1 D2 85Rb87Rb", "Strong quantum correlations in four wave mixing Glorieux2010", "Measurement of intensity difference squeezing Yu Meng Zhang2013"],
    "added_ids": [r["paper_id"] for r in records],
    "screened": [
        {"title":"Measurement of intensity difference squeezing via non-degenerate four-wave mixing process in an atomic vapor","url":"https://doi.org/10.1088/1674-1056/22/9/094204","decision":"candidate_full_text_access_gap","reason":"Relevant85RbD1 experiment. PublisherPDF endpoint redirected to an empty response; institutionalPDF returned403. Browser fallback on2026-10-23: official article showed abstract/references, upper and sidebarPDF buttons did not open a document, ArticleText expanded without a body; direct author-institutionPDF rendered only an empty dark canvas. Abstract alone reports−3.7dB with noisy seed but is insufficient to build the full operating-point record."},
        {"title":"Generating quantum correlated twin beams by four-wave mixing in hot cesium vapor","url":"https://doi.org/10.1103/PhysRevA.96.043843","decision":"excluded","reason":"Cesium, not85RbD1."},
        {"title":"Compact sub-kilohertz low-frequency quantum light source based on four-wave mixing in cesium vapor","url":"https://doi.org/10.1364/OL.43.001243","decision":"excluded","reason":"Cesium, not85RbD1."},
        {"title":"Time-resolved detection of relative-intensity squeezed nanosecond pulses in an87Rb vapor","url":"https://doi.org/10.1088/1367-2630/13/4/043030","decision":"excluded","reason":"FWM medium87Rb, not85Rb."},
        {"title":"Quantum correlations by four-wave mixing in an atomic vapor in a nonamplifying regime: Quantum beam splitter for photons","url":"https://doi.org/10.1103/PhysRevA.84.053826","decision":"future_extension","reason":"Relevant2011 full paper from the sameParis apparatus asGlorieux2010; avoid treating its9.2-dB headline as an independent laboratory holdout. Not extracted in this pass."},
        {"title":"Degenerate four-wave-mixing as a low-power source of squeezed light","url":"https://doi.org/10.1364/OE.411562","decision":"excluded","reason":"Degenerate/theoretical configuration differs from the requested nondegenerate double-lambda IDS source."}
    ],
    "notes": ["Source availability and values were checked on the recorded date; this is a bounded literature extension, not an exhaustive systematic review.","All inspected papers are a development literature pool, not an untouched future external validation set."]
}
(ROOT / "search_new.json").write_text(json.dumps(screen, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
print("Wrote",len(records),"new paper records;",sum(len(r['experiments']) for r in records),"experiment records")

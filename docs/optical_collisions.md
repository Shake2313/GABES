# Optical collision reference

OD, SAS, Lambda and unified Magneto can use measured Ne optical coefficients as
an opt-in reference. Enter the Ne partial pressure at the operating condition.
The reported optical frequency shift is the change in atomic resonance
frequency. The solver uses laser detuning minus that shift. For Lambda, the
common optical shift cancels in the Raman detuning. The reported elastic
optical decoherence rate is half the added homogeneous FWHM. The reference
keeps natural radiative decay and branching separate from elastic collisions.

## Sources and scope

- Rb D1 and D2 use Sargsyan et al., Figs. 3-6, including isotope-resolved
  measurements of both Rb isotopes with nominal Ne pressures 0, 6 and 20 Torr
  and a 120 C reservoir in a half-wave nanocell. The width is 10(1) MHz/Torr
  on both lines and the signed shifts are -1.1(2) and -2.1(2) MHz/Torr.
  Transferring this reference to bulk cells or other temperatures requires
  validation. [Sargsyan et al. 2021](https://doi.org/10.1134/S0030400X2108018X).
- Comparison slopes come from independent absorption/emission measurements.
  Rb D1 widths differ across conditions, including 12.2(97) MHz/Torr in
  Rb87 at 337 K. The same paper's microwave clock shifts concern a separate
  transition. [Ha et al. 2021](https://doi.org/10.1103/PhysRevA.103.022826),
  [Rotondaro and Perram 1997](https://doi.org/10.1016/S0022-4073(96)00147-1),
  [Ottinger et al. 1975](https://doi.org/10.1103/PhysRevA.11.1815).
- Cs133 D1 and D2 use Pitz et al.'s line-core absorption values at 313 K.
  D1 width/shift are 10.85(2)/-1.60(1), D2 9.81(6)/-2.58(1) MHz/Torr.
  Independent slopes retain the spread among methods and temperatures.
  [Cs D1](https://doi.org/10.1103/PhysRevA.80.062718),
  [Cs D2](https://doi.org/10.1103/PhysRevA.82.042502),
  [Bernabeu and Alvarez 1980](https://doi.org/10.1103/PhysRevA.22.2690),
  [Inoue et al. 1990](https://doi.org/10.1143/JPSJ.59.516).
- Rb self-broadening is density-linear. The Rb D1 coefficient preserves the
  established 0.69(4) x 10^-7 Hz cm^3 slope, measured up to 3e20 m^-3. The
  D2 reference has its own 1.10(17) x 10^-7 Hz cm^3 slope, measured by
  attenuated total reflection at 3.8e20 to 3.7e21 m^-3. Application below
  that range is an impact-regime extension. Cs self-broadening remains
  unsupported for both lines. [Weller et al. 2011](https://doi.org/10.1088/0953-4075/44/19/195006),
  [Kondo et al. 2006](https://doi.org/10.1103/PhysRevA.73.062504).

Uncertainty in each slope is the larger of the primary measurement uncertainty
and the largest root-sum-square of another measurement's reported uncertainty
and its difference from the primary value. This is a comparison bound, with
no Gaussian coverage claim. Linear propagation multiplies the bound by the
stated pressure. Pressure and temperature outside the source conditions are
identified in the report. Relative bounds above 20% are flagged. At zero
pressure both Ne optical terms are exactly zero. Unsupported coefficients are
reported explicitly. This model describes a Lorentzian impact line core.
Velocity-changing collisions, Dicke narrowing and microwave shifts need
separate data and models.

# Magneto buffer-cell ground relaxation

Choose **Buffer gas cell**, then **Quenching finite cylinder** under the
advanced ground relaxation control. Enter the operating Ne partial pressure,
inner cell radius, axial cell length, centered 1/e^2 beam radius and fully
quenching wall condition. The reference sets the single-region population
reset rate. The manual measured-rate control is available as a separate
choice. Incomplete or unsupported geometry reports insufficient calibration
and explicitly identifies the configured manual rate used for the spectrum.

## Model

A fully quenching wall sets the spin polarization to zero on the cylinder's
side and both endcaps. The slowest axially symmetric diffusion mode is
J0(j01 r/R) sin(pi z/L), with j01 = 2.4048255577. The rates D(j01/R)^2
and D(pi/L)^2 sum to the diffusion-limited wall loss in s^-1. The boundary
condition already accounts for quenching. The budget counts this loss once.
The beam radius enters a transport validity check and a beam-crossing time
diagnostic. It does not replace the cylinder radius in the lowest eigenmode.
A dimensionless proxy, (3D/mean Rb speed) divided by the smallest beam/cell
scale, must be at most 0.1. This screen excludes the zero-pressure and
near-ballistic limits. The model describes the long-time mode and does not
resolve the early-time illuminated-volume modes or a partly preserving wall.

The binary Rb-Ne spin-destruction proxy is n_Ne * sigma_sd * mean relative
speed. The model adds this term to the wall-loss rate in the single-region
isotropic reset. It leaves the separate configured ground pure dephasing
channel intact. The use of a spin-destruction cross section as an isotropic
reset is a scope approximation. The source does not establish a complete
ground-multipole or isotope-dependent collision operator.

## Sources and uncertainty

- Pouliot et al. Table II gives D = 0.214(14) cm^2/s at 297.15 K and 760 Torr
  for naturally abundant Rb-Ne. Table I uses D proportional to T^1.756 / p
  for Ne. Chrapkiewicz et al. give 0.18(3) at 273.15 K and 760 Torr. Franz
  reports 0.48(10) at 340.15 K and 760 Torr. These comparison measurements
  are rescaled to the same temperature and included in the uncertainty.
  The model uses the same trace-Rb diffusion coefficient for both Rb
  isotopes. [Pouliot et al. 2025](https://doi.org/10.1103/PhysRevA.111.033108),
  [Chrapkiewicz et al. 2014](https://doi.org/10.1007/s00340-013-5712-y),
  [Franz 1965](https://doi.org/10.1103/PhysRev.139.A603).
- The spin proxy uses the enriched 21Ne value 1.9(2) x 10^-23 cm^2 from
  Ghosh and Romalis (Ghosh dissertation Table 5.4). The older 3.3 x 10^-24
  and 5.2 x 10^-23 cm^2 spin-disorientation results bound the spread across
  source regimes. They do not supply fitted statistical errors for this
  term. The reported bound covers the largest deviation from the primary
  coefficient. [Ghosh and Romalis 2010](https://doi.org/10.1103/PhysRevA.81.043415),
  [Franz 1965](https://doi.org/10.1103/PhysRev.139.A603),
  [Franzen 1959](https://doi.org/10.1103/PhysRev.115.850).

For diffusion the bound includes measurement uncertainty and source spread.
The propagated radial and axial terms are fully correlated because both use
the same D. The total bound sums the wall and spin bounds without assuming
independent source errors. Apparatus measurement error is outside this
coefficient-only budget. No confidence level is assigned. Above 20%, the
derived rate stays opt-in and the readout reports the uncertainty.

## Validation

At D = 0.0018 m^2/s, R = 5 mm and L = 18 mm, independent radial finite
volumes and an axial finite difference converge to the analytic fundamental
within 0.486%, 0.132% and 0.0343% at 12, 24 and 48 samples. The axial-only
and radial-only limits preserve their expected inverse-square geometry
scaling. For Rb87-Ne at 20 Torr and 297.15 K, R = 7 mm, L = 20 mm and a 1 mm
beam radius, the reference gives 116.042 s^-1 of wall loss and 0.765 s^-1 of
spin destruction, total 116.808 s^-1. The coefficient-only bound is 85.9%.

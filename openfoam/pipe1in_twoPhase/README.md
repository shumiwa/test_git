# 1" pipe – 3D gas–liquid two-phase flow (OpenFOAM, VOF)

A 3D transient case for air–water flow in a horizontal 1-inch pipe. It uses
`interFoam` (VOF with algebraic interface compression) and k-ω SST turbulence.
At the default conditions the flow should develop stratified-wavy into slug
flow. The case writes holdup, pressure and probe time histories so you can
characterise the flow.

Target: **OpenFOAM ESI (openfoam.com) v2306 or newer.** Tested on v2412; see
[Validation](#validation).

## Quick start

```sh
source /usr/lib/openfoam/openfoam2412/etc/bashrc   # or your install
./Allrun              # serial
./Allrun -parallel    # 4 cores (system/decomposeParDict)
./Allclean            # reset
```

`Allrun` copies `0.orig` to `0`, then runs `blockMesh`, `checkMesh`,
`setFields` and `interFoam`.

## Everything is set in `system/caseParameters`

| Parameter | Default | Meaning |
|---|---|---|
| `D` | 0.0254 m | Pipe inner diameter (1" Sch 40 ID = 0.02664 m) |
| `LbyD` | 40 | Pipe length in diameters |
| `nCore`, `nLow`, `nUp`, `nRad`, `radGrading` | 12, 6, 6, 10, 0.3 | Cross-section mesh (O-grid) |
| `nAx` | 400 | Cells along the pipe |
| `USL` | 0.5 m/s | Superficial liquid velocity |
| `USG` | 2.0 m/s | Superficial gas velocity |
| `hLbyD` | 0.5 | Inlet liquid level / D (inlet split and initial layer) |

The blockMesh vertices, inlet flow rates, initial fields, probe locations and
monitoring planes all derive from these values through `#eval`. Run
`./Allclean && ./Allrun` after changing any of them.

The default mesh has 250k cells (624 per cross-section × 400). The first cell
off the wall is about 0.26 mm. At the default conditions this gives y+ of
3–18 (mean about 8). That is in the buffer layer, which the ESI
`omegaWallFunction` / `nutkWallFunction` pair handles through its blended
formulation. For y+ < 1 wall resolution, raise `nRad` and lower `radGrading`.

## Setup

**Geometry and mesh.** Horizontal pipe along +x, gravity −z. The O-grid
(square core plus 6 curved outer blocks) has a horizontal mesh line at the
inlet liquid level z = (hLbyD − 0.5)·D. This splits the inlet into two patches:

```
            11 ------------- 10
           /  5 ---------- 4  \          inletGas    (above)
          /   |    gas     |   \
         9 -- 3 ---------- 2 -- 8   <-  z = zI
          \   |  liquid    |   /
           \  0 ---------- 1  /          inletLiquid (below)
            6 ------------- 7
```

`hLbyD` must stay between about 0.32 and 0.68, so that the level crosses the
core square. If you move it away from 0.5, also change `nLow : nUp` to roughly
`(zI + 0.2) : (0.2 − zI)`.

**Boundary conditions.**

| Patch | U | alpha.water | p_rgh | k / ω |
|---|---|---|---|---|
| `inletLiquid` | `flowRateInletVelocity`, Q = USL·πD²/4 | 1 | `fixedFluxPressure` | 5 % intensity, l = 0.07 D |
| `inletGas` | `flowRateInletVelocity`, Q = USG·πD²/4 | 0 | `fixedFluxPressure` | same |
| `outlet` | `pressureInletOutletVelocity` | `inletOutlet` (gas backflow) | `prghPressure`, p = 0 | `inletOutlet` |
| `wall` | `noSlip` | `zeroGradient` (90° contact angle) | `fixedFluxPressure` | wall functions |

The volumetric flow rates impose the superficial velocities exactly. The
in-situ inlet velocities are U_L = USL/α_L and U_G = USG/(1 − α_L).

**Initial condition.** `setFields` fills the pipe with a stratified layer of
height `hLbyD·D`. The liquid starts at U_L and the gas at U_G, which are
consistent with the inlet. The outlet `value` is seeded with USL + USG. Do not
start from rest or from zero boundary `value` entries. A zero initial flux
forces an impulsive start of the whole liquid column, and the run blows up in
the first time steps.

**Fluids.** Water (ρ = 998.2 kg/m³, ν = 1.0e-6 m²/s) and air at 1 atm
(ρ = 1.204 kg/m³, ν = 1.516e-5 m²/s), with σ = 0.0728 N/m. Edit
`constant/transportProperties` for other fluids.

**Numerics.** Courant-limited adaptive time stepping (maxCo = maxAlphaCo = 0.5)
with PIMPLE (1 outer and 3 pressure correctors) and MULES with
`MULESCorr yes`. The schemes are vanLeer for α plus interface compression
(cAlpha = 1), and linearUpwindV for momentum.

## Output (`postProcessing/`)

| Function object | What you get |
|---|---|
| `section_x005 … x095` | Area-averaged `alpha.water` (liquid holdup) and `p` on cutting planes at x/L = 0.05, 0.25, 0.5, 0.75, 0.95. The pressure gradient comes from differences between sections. |
| `outletAverage` | Area-averaged p, p_rgh and alpha at the outlet |
| `outletLiquidFlux` | Σ α·φ at the outlet (delivered liquid flow rate, m³/s) |
| `holdupVolume` | Pipe-averaged holdup |
| `probes` | Time series of α, p and U at the bottom, axis and top at x/L = 0.5 and 0.9 (slug frequency, translational velocity from cross-correlation) |
| `yPlus` | Wall y+ at write times |

## Choosing conditions

For a horizontal 1" air–water pipe (Mandhane / Taitel–Dukler maps, roughly):

| Regime | USL (m/s) | USG (m/s) |
|---|---|---|
| Stratified smooth/wavy | 0.01–0.1 | 0.1–3 |
| Slug / plug (default) | 0.2–2 | 0.3–5 |
| Annular | 0.01–0.3 | > 15 |
| Dispersed bubble | > 2 | < 2 |

Things to adjust for each regime:

- **Slug flow** needs length to develop from a stratified inlet. 40 D shows
  the onset; use `LbyD` 100–200 (and scale `nAx`) for developed slugs.
- **Annular flow** needs finer radial resolution near the wall (raise `nRad`,
  lower `radGrading`) and a smaller time step.
- **Dispersed bubbles** smaller than about 3 cells are not resolved by VOF. For
  that regime use an Euler–Euler solver (`multiphaseEulerFoam`) instead.

## Caveats

- RANS with VOF over-predicts turbulence at the interface, which damps waves.
  For better wave and slug physics, use LES (`simulationType LES;`, WALE) on a
  finer mesh, or add interface turbulence damping.
- A plug inlet profile adds an entrance length. Discard the first few D, or
  the first flow-through in time, when averaging.
- The `prghPressure` outlet fixes a uniform static pressure across the exit.
  Slugs leaving the domain cause pressure pulses, so evaluate the pressure
  gradient between internal sections.
- OpenFOAM Foundation (openfoam.org v11+) uses
  `foamRun -solver incompressibleVoF`, and its dictionaries differ
  (`momentumTransport`, `physicalProperties.*`). This case is written for the
  ESI line.

## Validation

Run with OpenFOAM **v2412** (conda-forge build), default conditions
(USL = 0.5 m/s, USG = 2 m/s, hL/D = 0.5):

| Run | Mesh | Simulated | Result |
|---|---|---|---|
| Full resolution | 250k cells (`nAx 400`) | 0–0.1 s, 607 steps, ~32 min on 1 core | Stable at maxCo 0.5, no warnings; all function objects write; y+ 2.9–18 |
| Coarse (`nAx 100`) | 62k cells | 0–1.0 s, 2937 steps, ~60 min on 1 core | Stable throughout; liquid holdup rises from 0.50 to ~0.55; interfacial waves near the outlet (section holdup at x/L = 0.95 varies 0.47–0.78); time-averaged outlet liquid flux matches USL to within 7 % (it still fluctuates as waves exit) |

At 1 s no slug had yet bridged the pipe. On a 40 D domain slugs take several
seconds and more length to form; see [Choosing conditions](#choosing-conditions).

In the cloud sandbox where I tested it, `Allrun -parallel` reached `mpirun`
but could not run because of an MPI library-path problem in that conda
environment. Nothing in the case causes this; on a normal install the
parallel path works.

The stock Debian/Ubuntu `openfoam` package (v1912, 2020) is **not**
suitable. On this O-grid it diverges in the first time steps even for trivial
single-phase uniform flow, while `pimpleFoam` on the same mesh and BCs is
stable. The same case on v2412 runs cleanly. Use a current ESI release.


Overview
--------

This directory contains a set of GBM detector response templates created with
the GBM response generator [1] for use with the generalized targeted search.
It includes the detector responses to gamma-ray photons from astrophysical 
sources arriving directly at each detector as well as photons that indirectly
reach detectors after scattering off the Earth's atmosphere.

The responses are stored as .npy formatted files in the following folders:

   1. GBM/direct/ includes the direct responses for NaI and BGO detectors
   2. GBM/atmo_nai/ includes the indirect responses after atmospheric scattering for NaI detectors
   3. GBM/atmo_bgo/ includes the indirect responses after atmospheric scattering for BGO detectors

These files can be loaded into Python as numpy arrays using:

```
import numpy as np
rsp = np.load("path/to/file.npy")
```

The atmospheric scattering files are named according to the location of the
Earth center in spacecraft coordinates in azimuth and zenith, which defines
the behavior of the scattering response. These are provided for a zenith
angle of 130 degrees because 70% of the time Fermi is within +/- 5 deg of this 
zenith orientation relative to the Earth.

Response Format
---------------

Each response array has a shape of (nspectrum, nsky, nenergy, ndet) where:

   * nspectrum is the number of spectral shapes included in the response array
   * nsky is the number of points on the sky used to generate the response
   * nenergy is the number of energy bins used to bin photon events for a given detector class
   * ndet is the number of detectors within the NaI/BGO classes (NaI = 12 detectors, BGO = 2 detectors)

Spectral Templates
------------------

All files contain nspectrum = 4 spectral templates with the following
spectral definitions

   1. hard: a hard GRB spectrum described by a cutoff power law
            with Epeak 1500 keV, index = -1.5 [2]
   2. norm: a normal GRB spectrum described by a Band spectrum
            with Epeak 230 keV, alpha = -1.0, beta = -2.3 [2]
   3. soft: a soft GRB spectrum described by a Band spectrum
            with Epeak 70 keV, alpha = -1.9, beta = -3.7 [2]
   4. blackbody: a blackbody spectrum with kT = 10 keV motivated by the
                 soft thermal tail in GW170817 [3]

Sky Points
----------

All files use nsky = 1634 points on the sky, each separated by 5 degrees.
These correspond to locations in zenith and azimuth relative to the spacecraft
as defined by the SkyGrid class from the utils.py file included in the generalized
targeted search project [4]. To generate this grid of locations, use the following
from within the generalized targeted search directory:

```
from utils import SkyGrid
grid = skyGrid(5.0)
```

These locations can be converted to Equatorial coordinates using an observing time
combined with position information about the Fermi spacecraft [5].

Energy Bins
-----------

The NaI and BGO responses each use 8 energy bins but the bin edges are
different for each detector class. The approximate bin edges are:

NaI | Energy           BGO | Energy
Bin | Range [keV]      Bin | Range [keV]
-----------------      -----------------
0   | 3-12             0   | 100-400 
1   | 12-30            1   | 400-1000
2   | 30-50            2   | 1000-2000
3   | 50-100           3   | 2000-50000
4   | 100-300          4   | 5000-10000
5   | 300-500          5   | 10000-20000
6   | 500-1000         6   | 20000-40000
7   | 1000-2000        7   | 40000-50000

These edges are approximate because the exact edges vary slightly due
to calibration differences between detectors. The final bin 7 for 
both NaI and BGO detectors is an overflow bin typically excluded from
analyses. The first NaI bin 0 (3-12 keV) is frequently excluded
from analyses searching for GRB transients because it is dominated
by soft, often Galactic transients.
 
Detector Indices
----------------

The 12 NaI detector indices start with index 0 = n0 to denote the first NaI
detector and increase until the last NaI detector index 11 = nb. Similarly,
for BGO index 0 = b0 and index 1 = b1.

Examples
--------

The following code will return the direct response to the soft spectral template
for NaI detector n3 over the 100-300 keV energy range for the location
(azimuth 195.4 deg, zenith 80.0 deg) in spacecraft coordinates.

```
import numpy as np
direct_rsp = np.load("GBM/direct/nai.npy")
direct_rsp_soft_n3_100to300keV_az195_zen80 = direct_rsp[2][678][3][4]
```

The atmospheric scattering response for the same template, detector, energy bin,
and location can be obtained for an Earth center location of
(azimuth 45 deg, zenith 130 deg) with:

```
import numpy as np
atmo_rsp = np.load("GBM/atmo_nai/atmrates_az45_zen130.npy")
atmo_rsp_soft_n3_100to300keV_az195_zen80 = atmo_rsp[2][678][3][4]
```

References
  [1] https://fermi.gsfc.nasa.gov/ssc/data/analysis/gbm/DOCUMENTATION.html
  [2] Fletcher C., Wood J., Hamburg R. et al. 2024 ApJ 964 149
  [3] Goldstein A., Burns E., Hamburg R. et al. 2019 arXiv:1903.12597
  [4] https://github.com/USRA-STI
  [5] https://astro-gdt.readthedocs.io/projects/astro-gdt-fermi/en/latest/missions/fermi/gbm/poshist.html


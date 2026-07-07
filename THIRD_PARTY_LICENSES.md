# Third-Party Licenses

FullFlow depends on or integrates with the following third-party projects and
libraries.  Users are responsible for complying with all licenses that apply to
their use, modification, or redistribution of FullFlow and its dependencies.

## FullPlot

FullPlot is used for trace objects, command traces, sensor traces, map-generation
workflows, and HDF5 plotting/inspection workflows.

Project:
https://github.com/saakethramoju/FullPlot

Package:
https://pypi.org/project/fullplot/

License:
See the FullPlot project license.

---

## ThermoProp

ThermoProp is an optional integration used by many FullFlow examples for fluid,
material, propellant, and chemical-equilibrium properties.  FullFlow's core
package does not import ThermoProp at top level, but users can install
`fullflow[thermo]` or `fullflow[examples]` for ThermoProp-backed workflows.

Project:
https://github.com/saakethramoju/ThermoProp

Package:
https://pypi.org/project/thermoprop/

License:
ThermoProp is released under the GNU General Public License version 3.

---

## NumPy

NumPy is used for numerical arrays and scalar/vector calculations.

Project:
https://numpy.org/

License:
BSD 3-Clause License.

---

## SciPy

SciPy is used for nonlinear least-squares solving and interpolation utilities.

Project:
https://scipy.org/

License:
BSD 3-Clause License.

---

## Rich

Rich is used for optional terminal diagnostics and solver tables.

Project:
https://github.com/Textualize/rich

License:
MIT License.

---

## h5py

h5py is used for HDF5 export and result storage.

Project:
https://www.h5py.org/

License:
BSD-style license.

---

## Optional and indirect dependencies

When users install optional ThermoProp-backed workflows, they may also interact
with dependencies documented by ThermoProp, including property packages such as
CoolProp, PYroMat, and RocketProps.  Refer to ThermoProp's third-party license
file for details.

---

## Disclaimer

FullFlow is an independent project and is not affiliated with, endorsed by, or
sponsored by FullPlot, ThermoProp, NumPy, SciPy, h5py, Rich, CoolProp, PYroMat,
RocketProps, or any other third-party dependency.  All trademarks, copyrights,
and licenses remain the property of their respective owners.

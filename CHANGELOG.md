# Changelog

## 2.0.1

### PID controller

* Reworked PID integral action as a normal public FullFlow dynamic State instead
  of advancing controller history inside repeated nonlinear residual evaluations.
* Added ``integral_error`` and ``integral_error_dot`` as the PID dynamic pair so
  the implicit transient solver integrates accumulated error with the rest of the
  network.
* Changed derivative history to use the previous accepted ``error`` and network
  time through the public ``State.previous`` interface.
* Made transient takeover bumpless whether ``trim`` is omitted or supplied.
  The initialized command now defines the first PID output in both cases.
* Kept ``trim`` as a live feed-forward contribution after startup, so later trim
  changes still move the requested command immediately.
* Improved conditional-integration anti-windup so it considers the direction of
  the integral contribution and supports either sign of integral gain.
* Added public controller diagnostics for proportional, integral, derivative,
  raw command, saturation status, and startup command bias.
* Added clear validation for reversed command limits and derived, non-writable
  command States.
* Kept the PID transient-only so steady-state solves preserve the user-established
  plant command and the controller takes over when transient integration begins.
* Did not add or remove PID constructor arguments.

### Sequence examples

* Replaced hidden mutable closure state in the bang-bang tank main-valve example
  with an accepted-timestep Sensor-triggered command for the one-time opening
  event.
* Updated bang-bang and relief-valve hysteresis examples to retain their previous
  accepted command through explicit State inputs rather than hidden Python
  closure variables.
* Preserved repeated relief-valve open/close/reopen behavior within one transient
  simulation.
* Added comments explaining when to use a one-time Sensor command versus
  repeatable State-based hysteresis.

## 2.0.0

FullFlow 2.0.0 is a publish-ready public release focused on packaging readiness,
repository documentation, public API documentation, solver documentation, and
basic release validation.

### Packaging

* Bumped package version to `2.0.0`.
* Replaced the previous local development dependency on FullPlot with the PyPI dependency `fullplot>=0.1.0`.
* Updated PyPI metadata, keywords, classifiers, and package description.
* Added wheel license inclusion through `license-files = ["LICENSE"]`.
* Added optional dependency groups:
  * `thermo` for ThermoProp-backed examples and workflows.
  * `examples` for the full example dependency set.
  * `dev` for testing and release checks.
* Added source-distribution inclusion rules for README, changelog, license files, publishing notes, examples, tests, and docs.
* Cleaned generated files, macOS resource-fork files, `__pycache__`, `.pyc`, `.DS_Store`, virtual-environment artifacts, and build output from the publish-ready archive.
* Updated the PyPI release workflow so tests run before build and publish.

### Documentation

* Rewrote `README.md` as the primary user guide because FullFlow does not yet have a separate official documentation site.
* Added detailed README sections covering:
  * Installation and optional extras.
  * Dependency model and the new PyPI FullPlot dependency.
  * Core concepts: `Network`, `State`, `Component`, `Balance`, and `Model`.
  * Static evaluation, steady-state solving, transient solving, quasi-steady sweeps, and HDF5 output.
  * Every public solver option for `SteadyState.solve(...)` and `Transient.solve(...)`.
  * Component authoring and dynamic-equation conventions.
  * Component catalog grouped by flow, compressible flow, nodes/storage, heat transfer, convection, friction factors, turbomachinery, propulsion, maps, lookups, sensors, sequences, PID controllers, and actuators.
  * Lookup, map, sensor, sequence, controller, and actuator examples.
  * Units, debugging workflow, and development checks.
* Expanded docstrings for core public classes including `State`, `StateLike`, `Balance`, `Component`, `Network`, and `Model`.
* Expanded docstrings for the public solver front ends `SteadyState` and `Transient`.
* Added or expanded component docstrings for all currently exported components:
  * General flow: `FlowTube`, `AdiabaticFlow`, `DarcyWeisbach`, `DischargeCoefficient`, `CavitatingVenturi`, `SeriesCdA`, `ParallelCdA`, `RectanglePoiseuille`, `EllipsePoiseuille`, `CircularAnnulusPoiseuille`, and `HydraulicDiameter`.
  * Compressible flow: `CompressibleOrifice`, `IsentropicDiffuser`, and `IsentropicNozzle`.
  * Nodes/storage: `Volume`, `Solid`, and `Composition`.
  * Heat transfer: `Conduction`, `Convection`, `Radiation`, `AmbientRadiation`, `TemperatureRecoveryFactor`, `AdiabaticWallTemperature`, and `EckertReferenceTemperature`.
  * Convection coefficients: `Gnielinski`, `Miropolskii`, `Petukhov`, `SiederTate`, `DittusBoelter`, `Bartz`, `NaturalConvection`, and `ChurchillChu`.
  * Friction factors: `Colebrook`, `Churchill`, and `PetukhovFriction`.
  * Turbomachinery and propulsion: `Rotor`, `GasTurbine`, `ConstantDensityPump`, `PolytropicPump`, `SpecificImpulse`, and `IdealCharacteristicVelocity`.
  * Data/control/instrumentation: `Lookup`, `LookupAttribute`, `Map`, `Sensor`, `SensorEvent`, `SensorCondition`, `Sequence`, `SequenceCommand`, `SequenceCondition`, `SequenceAbort`, `PID`, and `Actuator`.
* Added detailed docstrings for `Component.setup(...)`, component initialization helpers, model-option helpers, and export-control hooks.
* Expanded FullFlow-specific exception docstrings so user-facing failures are easier to understand from `help(...)` and IDE inspection.
* Added `PUBLISHING.md` with release-check, build, artifact-inspection, and publishing instructions.
* Updated `THIRD_PARTY_LICENSES.md` to include FullPlot and the optional ThermoProp integration model.

### Reliability and readiness checks

* Ran compile checks for package source, tests, and examples.
* Ran the package test suite.
* Ran import and smoke checks for the public package API.
* Built wheel and source distribution artifacts.
* Inspected wheel and source distribution contents for license, README, changelog, docs, tests, examples, and package files.
* Performed an installed-wheel smoke test from the built artifact.

### Notes

* No component equations, solver equations, or physical correlations were intentionally changed for this release.
* This release is documentation- and packaging-heavy because the repository currently serves as the primary documentation source.


## 0.1.3

### System Core

* Added a shared `StateLike` protocol/helper layer so `State`, `CallableLookupAttribute`, and future state-compatible proxies are handled consistently without unsafe dynamic probing.
* Added a monotonic `Network.version` and public `Network.mark_dirty()` / `mark_structure_changed()` hooks so solver runtime metadata can be cached safely and invalidated only when network structure changes.
* Refactored `State`, `Composition`, `Balance`, `Component`, and `Network` for a smaller, faster core API.
* Removed the unused `Exceptions` package and replaced it with direct built-in exception types.
* Replaced NumPy usage in core containers with standard-library math where possible to reduce import overhead.
* Added cached constructor metadata in `Component.setup()` to avoid repeated signature inspection.
* Streamlined component setup so only declared constructor parameters become public component attributes.
* Added cached Network iteration-variable metadata, labels, bounds, and values with automatic invalidation when components or balances are added or removed.
* Preserved the existing readable `Network.iteration_variables` summary while adding `iteration_variable_labels`, `iteration_variable_summary`, and `iteration_variable_states` for diagnostics and solver internals.
* Centralized residual normalization and export serialization in `Network` to remove duplicated component/balance logic.
* Kept support for assignable state-like lookup attributes as iteration variables, including `CallableLookupAttribute` inputs such as `eq.pressure`.

### Solvers

* Added clean transient stopping for sensors with unavailable test data when `extend=False`.

* Refactored `SteadyState.py` into a thin public wrapper backed by the modular `Solvers/steady_state/` implementation.
* Added a network-version-aware runtime plan/cache for iteration variables, bounds, component evaluation callables, residual owners, and state-settling references.
* Reduced repeated network introspection inside residual evaluations and state-settling passes.
* Preserved existing `SteadyState(network).solve(...)` and `static_evaluate(...)` behavior.
* Added simpler call paths: `network.solve(...)`, `network.static_evaluate(...)`, `SteadyState(network).run(...)`, and `SteadyState(network)(...)`.
* Removed the placeholder `Transient.py` module.
* Hardened solver and network state discovery so dynamic `CallableLookup` objects are not accidentally treated as iterable compositions.


### Components

* Added active `Sensor` anchoring: sensors can now sample FullPlot Trace objects at solver time and expose a balance through `variable`/`data`. Missing data can either be extended through by holding the variable or stop a transient/forced-steady run cleanly.

* Streamlined scalar branch math used by discharge coefficients, regulators, nozzles, friction factors, pumps, and selected compressible-flow components. Common reversible-flow helpers now live in `Branches/_flow_math.py`, preserving existing equations while avoiding unnecessary NumPy scalar dispatch.
* Reduced component import overhead by keeping NumPy only where arrays/root solving are actually needed.
* Improved `CallableLookup` performance by caching dynamic attribute proxies, callable signatures, reusable-object update signature checks, and optional memoized lookup results via `memo_size`. Added the `lazy` alias for deferred lookup evaluation.
* Made ThermoProp-dependent combustion chamber utilities import ThermoProp lazily so importing `fullflow` does not fail in environments where optional ThermoProp objects are not being used.
* Added `Component` helper methods for custom components: `_iteration_variable_names`, `values(...)`, `value(...)`, `numeric(...)`, `residual(...)`, `assign(...)`, `assign_state(...)`, `make_state(...)`, and `iteration_states(...)`. These simplify user-written components without changing existing component physics.
* Added lower-case public re-export paths under `fullflow.core` and `fullflow.components.*` while preserving legacy imports from `fullflow.System.*`.

### Packaging

* Cleaned release artifacts by excluding `.git`, `.venv`, `__pycache__`, `__MACOSX`, and `.DS_Store` from generated source zips.

### Documentation

* Expanded and standardized NumPy-style docstrings throughout FullFlow.

* Added detailed parameter documentation across components, including:

  * Inputs
  * Outputs
  * Units
  * Optional parameters
  * Solver-related variables

* Added governing-equation documentation for fluid, thermal, and network components.

* Added residual-equation documentation for iterative components.

* Added iteration-variable documentation where applicable.

* Improved documentation consistency across:

  * Fluid-flow components
  * Heat-transfer components
  * Thermal-network components
  * Solver utilities

* Improved readability of component documentation for both users and developers.

* Improved MkDocStrings compatibility and API-reference generation.

* Improved rendering of parameter and output tables within generated documentation.

* Improved consistency between source-code documentation and published API references.

* Expanded inline engineering notes and implementation references throughout the codebase.

### Improved

* Improved project documentation structure and navigation.

* Improved generated API-reference quality.

* Improved maintainability of component documentation.

* Improved consistency of engineering terminology and units throughout the library.

* Improved support for future automated documentation generation and website expansion.

## 0.1.2

### Documentation

* Added comprehensive NumPy-style docstrings throughout FullFlow components.
* Standardized component documentation format across the library.
* Added governing equations, residual definitions, iteration variable descriptions, and parameter documentation where applicable.
* Improved MkDocStrings compatibility for automatic API reference generation.
* Improved generated documentation readability and consistency.

## 0.1.1

### Added

* Added example gallery to the project documentation.
* Added pump-fed rocket engine example.
* Added mixture splitter example.
* Added counterflow heat exchanger example.
* Added model comparison and correlation sweep example.
* Added project metadata, keywords, classifiers, and URLs.
* Added Pandas dependency for DataFrame-based solution export.
* Added OpenPyXL dependency for Excel export support.
* Added GitHub Actions workflow for automated PyPI releases.

### Improved

* Improved package structure and import behavior.
* Improved documentation and project presentation.
* Improved README with feature overview, applications, and usage examples.
* Improved PyPI metadata and package discoverability.
* Improved dependency management and release workflow.

## 0.1.0

### Added

* Initial FullFlow package structure.
* Added System package.
* Added Solvers package.
* Added Exceptions package.
* Added Utilities package.
* Added uv-based build configuration.

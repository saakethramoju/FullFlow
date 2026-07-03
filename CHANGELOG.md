# Changelog

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

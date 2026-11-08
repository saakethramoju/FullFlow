# Lookup integration

`Lookup` connects ordinary Python functions, classes, or property packages to
FullFlow `State` objects. It is useful when properties must change with pressure,
temperature, composition, or other coupled network variables without exposing
those dependencies as a custom physical `Component`.

## Function outputs

```python
from fullflow import Lookup, Network, State

network = Network("Gas Properties")
pressure = State(2.0e5)
temperature = State(300.0)

def ideal_gas(pressure, temperature):
    return {"density": pressure / (287.0 * temperature)}

gas = Lookup(
    "Air", network, ideal_gas,
    pressure=pressure, temperature=temperature,
    outputs=("density",),
)

density = gas.density
```

`gas.density` is a state-like `LookupAttribute`. It can be used directly in
component connections, expressions, balances, and tracked outputs. The wrapped
callable receives the currently resolved values of its input States, not the
State objects themselves, unless the callable deliberately expects another
kind of object.

## Chaining and reusable objects

A Lookup may wrap a constructor whose returned object exposes properties.
Another Lookup can consume that object directly:

```python
reactants = Lookup(
    "Reactants", network, Reactants,
    fuels=fuel, oxidizers=oxidizer, mixture_ratio=mixture_ratio,
)

equilibrium = Lookup(
    "Equilibrium", network, Equilibrium,
    reactants=reactants, pressure=chamber_pressure,
)
```

Class instances that support updating their inputs can be reused rather than
reconstructed when possible. Input values are resolved recursively, so a
change in an upstream Lookup can propagate through chained Lookups during the
same network-evaluation process.

For property models with alternative independent-variable pairs, `priority`
selects the first available accepted input within each group. The longer
priority and initialization examples are in `README.md` and
`examples/general/7lookup_class_chaining.py`.

## Caching and solver behavior

`Lookup` caches resolved inputs, can reuse a wrapped object, and memoizes recent
results to avoid unnecessary external-library work. These mechanisms are
automatic; no special adapter is required for individual property packages.
For an expensive callable, the dominant cost can still be re-evaluating it
with *different* trial inputs during numerical Jacobian construction.

Callable outputs should be deterministic at identical inputs. Do not mutate
external state in a way that makes a nonlinear residual depend on evaluation
order. `Lookup.help()`, `.wrapped_signature`, `.wrapped_doc`, and `.obj` are
available when inspecting the wrapped API or current object.

See `examples/general/6lookup_function.py` and
`examples/general/7lookup_class_chaining.py`.

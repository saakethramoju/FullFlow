"""
Multiple Balance example.

A network can contain more than one Balance. Each Balance contributes one
algebraic equation. Usually, each equation also needs one solve variable.

This example solves two unknowns, x and y, from two equations:

    x + y = 10
    x*y   = 24

The first residual is written directly as a State expression. The second
residual is written as a normal Python function. Both styles are useful.
"""

from fullflow import *


EquationNetwork = Network("Multiple Balance Example")

x = State(2.0)
y = State(8.0)

sum_target = State(10.0)
product_target = State(24.0)

# A residual can be a State expression.
sum_residual = x + y - sum_target

# A residual can also be a function. This is useful when the equation is easier
# to read with normal Python code.
def product_residual():
    return x.value * y.value - product_target.value

SumBalance = Balance(
    "Sum Balance",
    EquationNetwork,
    variable=x,
    function=sum_residual,
)

ProductBalance = Balance(
    "Product Balance",
    EquationNetwork,
    variable=y,
    function=product_residual,
    bounds=(0.0, None),
)

EquationNetwork.track("x", x)
EquationNetwork.track("y", y)
EquationNetwork.track("x + y", x + y)
EquationNetwork.track("x*y", x * y)

SteadyState(EquationNetwork).solve(verbose=True)

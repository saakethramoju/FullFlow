_CEA_CACHE = {}

PROPELLANT_NAME_BANK = {
    # -------------------------
    # OXIDIZERS
    # -------------------------
    "lox": "LOX",
    "liquid oxygen": "LOX",
    "oxygen": "LOX",
    "o2": "LOX",
    "ox": "LOX",

    "n2o": "N2O",
    "nitrous": "N2O",
    "nitrous oxide": "N2O",

    "h2o2": "H2O2",
    "peroxide": "H2O2",
    "hydrogen peroxide": "H2O2",

    "f2": "F2",
    "fluorine": "F2",

    # -------------------------
    # FUELS
    # -------------------------

    # RP-1 / Kerosene family
    "rp-1": "RP-1",
    "rp1": "RP-1",
    "rocket propellant-1": "RP-1",
    "kerosene": "RP-1",
    "jet-a": "RP-1",
    "jeta": "RP-1",
    "n-dodecane": "RP-1",
    "dodecane": "RP-1",
    "ndodecane": "RP-1",

    # Hydrogen
    "lh2": "LH2",
    "liquid hydrogen": "LH2",
    "hydrogen": "LH2",
    "h2": "LH2",

    # Methane
    "methane": "CH4",
    "ch4": "CH4",
    "lng": "CH4",

    # Propane
    "propane": "C3H8",
    "c3h8": "C3H8",
    "n-propane": "C3H8",
    "npropane": "C3H8",

    # Ethanol
    "ethanol": "C2H5OH",
    "alcohol": "C2H5OH",
    "c2h5oh": "C2H5OH",

    # IPA
    "isopropanol": "C3H8O",
    "ipa": "C3H8O",

    # Hydrazine family
    "hydrazine": "N2H4",
    "n2h4": "N2H4",

    "mmh": "MMH",
    "monomethylhydrazine": "MMH",

    "udmh": "UDMH",
    "unsymmetrical dimethylhydrazine": "UDMH",

    # Ammonia
    "nh3": "NH3",
    "ammonia": "NH3",
}





def normalize_propellant_name(name: str) -> str:
    """
    Convert arbitrary user input into a RocketCEA-compatible propellant name.

    Parameters
    ----------
    name : str
        Input propellant name (case-insensitive, flexible aliases)

    Returns
    -------
    str
        RocketCEA-compatible propellant name

    Raises
    ------
    ValueError
        If the propellant is not recognized
    """
    if not isinstance(name, str):
        raise TypeError("Propellant name must be a string")

    key = name.strip().lower()

    if key in PROPELLANT_NAME_BANK:
        return PROPELLANT_NAME_BANK[key]

    raise ValueError(
        f"Unknown propellant '{name}'. Add it to PROPELLANT_NAME_BANK."
    )




def create_SI_CEA_object(fuel: str = 'RP-1', oxidizer: str = 'LOX'):
    """
    Create and return a RocketCEA `CEA_Obj` configured with SI-based units.

    This helper function initializes a `CEA_Obj` with consistent unit settings
    for propulsion analysis workflows. It is intended for use in chamber,
    injector, and nozzle performance calculations (e.g., c*, Isp, gamma).

    The returned object uses:
        - Temperature: Kelvin
        - c*: m/s
        - Sonic velocity: m/s
        - Enthalpy: J/kg
        - Density: kg/m^3
        - Specific heat: kJ/kg-K
        - Pressure: Pa

    Notes
    -----
    RocketCEA internally expects chamber pressure in psia by default.
    Ensure proper unit conversion when passing pressure values.

    FAC (Finite Area Combustor) support may be added in future versions.

    Parameters
    ----------
    fuel : str, optional
        Fuel propellant name (must match a valid RocketCEA fuel identifier).
        Default is 'LOX'.
    oxidizer : str, optional
        Oxidizer propellant name (must match a valid RocketCEA oxidizer identifier).
        Default is 'RP-1'.

    Returns
    -------
    CEA_Obj
        Configured RocketCEA object ready for thermochemical property evaluation.

    Raises
    ------
    ValueError
        If the specified fuel or oxidizer name is not recognized by RocketCEA.

    Examples
    --------
    >>> cea = create_CEA_object('RP-1', 'LOX')
    >>> cstar = cea.get_Cstar(Pc, MR)
    """
    from rocketcea.cea_obj_w_units import CEA_Obj
    key = (fuel, oxidizer)
    if key not in _CEA_CACHE:
        _CEA_CACHE[key] = CEA_Obj(
            oxName=normalize_propellant_name(oxidizer),
            fuelName=normalize_propellant_name(fuel),
            temperature_units='degK',
            cstar_units='m/sec',
            specific_heat_units='kJ/kg degK',
            sonic_velocity_units='m/s',
            enthalpy_units='J/kg',
            density_units='kg/m^3',
            pressure_units='Pa'
        )

    return _CEA_CACHE[key]
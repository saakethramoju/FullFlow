from fullflow import *
from thermoprop import *
import fullplot as fplt
import math









psia_to_pa = 6894.76

data = fplt.open("hotfire.h5")
time = data.time("time")
time.zero_at(1)




ftpt = data.trace(y="FTPT", x=time, name='Fuel Tank Pressure').window(start=0, stop=10) * psia_to_pa
ftpt_filt = ftpt.filter("moving_average", window=0.05)
fflow = data.trace(y="FUEL_MDOT", x=time, name='Fuel Mass Flow')
fipt = data.trace(y="FIPT", x=time, name='Fuel Injector Pressure').window(start=0, stop=10) * psia_to_pa

fflow_redline = fplt.Trace.constant("FFLOW Minimum Flow Redline", x=time, y=0.2, role='redline')
fflow_yellowline = fplt.Trace.constant("FFLOW Minimum Flow Warning", x=time, y=0.3, role='yellowline')

data.plot(
    y=[ftpt_filt / psia_to_pa, fipt / psia_to_pa],
    y2=[fflow, fflow_redline],
    xlabel='Time [s]',
    ylabel='Pressure [psia]',
    y2label="Mass Flow [kg/s]"
)
#data.plot(fflow)


Test = Network("Test")


Fuel = Lookup(
    "Fuel",
    Test,
    Propellant,
    "rp-1",
    pressure=ftpt_filt(0), # make sure it's using the correct state
    temperature=298.15
)

Line = DischargeCoefficient(
    "Line",
    Test,
    upstream_pressure=Fuel.pressure,
    downstream_pressure=fipt(0),
    density=Fuel.density,
    discharge_coefficient=1,
    cross_sectional_area=(math.pi/5) * (0.5/39.37)**2,
    #length=3,
    mass_flow=fflow(0.0)
)


FIPT = Sensor(
    "FIPT",
    Test,
    reading=Line.downstream_pressure,
    variable=Line.downstream_pressure,
    data=fipt,
)


FTPT = Sensor(
    "FTPT",
    Test,
    reading=Line.upstream_pressure,
    variable=Line.upstream_pressure,
    data=ftpt_filt,
)


FFLOW = Sensor(
    "FFLOW",
    Test,
    reading=Line.mass_flow,
    variable=Line.discharge_coefficient,
    data=fflow,
    conditions=[fflow_redline, fflow_yellowline]
)


SteadyState(Test).solve(
    dt=0.01,
    t_final=15,
    verbose=True,
    filename='test',
    #statistics=True
)

result = fplt.open("test.h5").at("Test/transient/runs/base/components")
result.tree()


result.plot(
    x="time",
    y="Line/discharge_coefficient",
    y2="Line/mass_flow",
    xlabel="Time [s]",
    ylabel="Discharge Coefficient",
    y2label="Mass Flow [kg/s]",
    title="Line",
)


fplt.show()



"""
Create a synthetic hotfire HDF5 file with many realistic sensor traces.

The output layout is intentionally simple:

    synthetic_hotfire.h5
        time
        PCMC_1
        PCMC_2
        PCMC_3
        PBTC_1
        PBTC_2
        SHAFT_RPM_1
        ...
        MOV_CMD
        MFV_CMD

This is meant to behave like a converted test-data file where every channel is
just a normal HDF5 dataset. Units are included only as optional dataset
attributes. FullPlot does not need them.
"""

from pathlib import Path

import h5py
import numpy as np


filename = Path("hotfire.h5")

rng = np.random.default_rng(7)

# ---------------------------------------------------------------------------
# Time base
# ---------------------------------------------------------------------------

dt = 0.005                    # 200 Hz sample rate
time = np.arange(-5.0, 20.0 + dt, dt)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def smooth_step(t, start, stop):
    """
    Smooth transition from 0 to 1 between start and stop.
    """
    x = np.clip((t - start) / (stop - start), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def pulse_train(t, start, stop, level=1.0):
    """
    Smooth-ish on/off command pulse.
    """
    return level * (smooth_step(t, start, start + 0.05) - smooth_step(t, stop, stop + 0.05))


def first_order_lag(input_signal, tau, dt):
    """
    Simple first-order sensor/system lag.
    """
    output = np.empty_like(input_signal)
    output[0] = input_signal[0]

    alpha = dt / (tau + dt)

    for i in range(1, len(input_signal)):
        output[i] = output[i - 1] + alpha * (input_signal[i] - output[i - 1])

    return output


def add_noise(signal, sigma, drift=0.0, spike_probability=0.0, spike_sigma=0.0):
    """
    Add white noise, slow drift, and occasional spikes.
    """
    n = len(signal)

    noisy = signal.copy()
    noisy += rng.normal(0.0, sigma, n)

    if drift != 0.0:
        drift_trace = np.cumsum(rng.normal(0.0, drift, n))
        noisy += drift_trace

    if spike_probability > 0.0:
        spikes = rng.random(n) < spike_probability
        noisy[spikes] += rng.normal(0.0, spike_sigma, spikes.sum())

    return noisy


def write_channel(h5, name, data, units=None, description=None):
    """
    Write a 1D channel to the HDF5 file.
    """
    dataset = h5.create_dataset(name, data=np.asarray(data), compression="gzip", shuffle=True)

    if units is not None:
        dataset.attrs["units"] = units

    if description is not None:
        dataset.attrs["description"] = description

    return dataset


# ---------------------------------------------------------------------------
# Command traces
# ---------------------------------------------------------------------------

PURGE_CMD = pulse_train(time, -4.5, 14.0, level=1.0)
IGN_CMD = pulse_train(time, 0.0, 0.6, level=1.0)

MOV_CMD = pulse_train(time, -0.30, 12.00, level=1.0)
MFV_CMD = pulse_train(time, 0.05, 11.85, level=1.0)

GG_MOV_CMD = pulse_train(time, -0.60, 12.15, level=1.0)
GG_MFV_CMD = pulse_train(time, -0.45, 12.05, level=1.0)

VENT_CMD = pulse_train(time, 12.1, 18.0, level=1.0)


# ---------------------------------------------------------------------------
# Base engine operating envelope
# ---------------------------------------------------------------------------

start_ramp = smooth_step(time, 0.05, 0.55)
shutdown_ramp = 1.0 - smooth_step(time, 11.8, 12.4)
mainstage = start_ramp * shutdown_ramp

# Small low-frequency throttle movement during mainstage.
throttle = 1.0 + 0.018 * np.sin(2.0 * np.pi * 0.32 * time) * mainstage
throttle += 0.006 * np.sin(2.0 * np.pi * 1.20 * time + 0.5) * mainstage

# Ignition transient overshoot.
ignition_bump = 1.0 + 0.18 * np.exp(-((time - 0.42) / 0.13) ** 2)

# Shutdown tail.
shutdown_tail = np.exp(-np.maximum(time - 12.2, 0.0) / 0.65)


# ---------------------------------------------------------------------------
# Chamber pressure traces, psia
# ---------------------------------------------------------------------------

pc_nominal = 14.7 + mainstage * (300.0 * throttle * ignition_bump)
pc_nominal += (time > 12.2) * 35.0 * shutdown_tail
pc_nominal = first_order_lag(pc_nominal, tau=0.025, dt=dt)

# Combustion roughness/ripple.
roughness = mainstage * (
    2.0 * np.sin(2.0 * np.pi * 180.0 * time)
    + 0.8 * np.sin(2.0 * np.pi * 74.0 * time + 1.1)
)

PCMC_1 = add_noise(pc_nominal + roughness, sigma=0.85, drift=0.0005, spike_probability=0.0006, spike_sigma=10.0)
PCMC_2 = add_noise(first_order_lag(pc_nominal * 0.997 + roughness * 0.7, tau=0.035, dt=dt), sigma=1.10, drift=0.0004)
PCMC_3 = add_noise(first_order_lag(pc_nominal * 1.006 + roughness * 0.5, tau=0.045, dt=dt), sigma=0.95, drift=0.0007)
PCMC_4 = add_noise(first_order_lag(pc_nominal * 1.002, tau=0.060, dt=dt), sigma=1.40, spike_probability=0.0004, spike_sigma=18.0)


# ---------------------------------------------------------------------------
# Injector / manifold / tank pressures, psia
# ---------------------------------------------------------------------------

OIPT = 14.7 + mainstage * (390.0 * throttle) + 20.0 * smooth_step(time, -3.8, -1.2)
FIPT = 14.7 + mainstage * (360.0 * throttle) + 15.0 * smooth_step(time, -3.5, -1.0)

OIPT += mainstage * 4.0 * np.sin(2.0 * np.pi * 34.0 * time)
FIPT += mainstage * 3.2 * np.sin(2.0 * np.pi * 29.0 * time + 0.7)

OIPT = add_noise(first_order_lag(OIPT, tau=0.05, dt=dt), sigma=1.0, drift=0.0004)
FIPT = add_noise(first_order_lag(FIPT, tau=0.05, dt=dt), sigma=1.0, drift=0.0004)

OX_MAN_1 = add_noise(OIPT - mainstage * 45.0 + rng.normal(0, 0.2), sigma=0.8)
OX_MAN_2 = add_noise(OIPT - mainstage * 48.0, sigma=0.9)
FUEL_MAN_1 = add_noise(FIPT - mainstage * 35.0, sigma=0.8)
FUEL_MAN_2 = add_noise(FIPT - mainstage * 38.0, sigma=0.9)

# Tank pressures are intentionally higher than the injector pressures so they
# behave like upstream supply pressures.
OTPT = add_noise(520.0 + 20.0 * smooth_step(time, -4.0, -1.0) - 18.0 * mainstage, sigma=0.8)
FTPT = add_noise(485.0 + 16.0 * smooth_step(time, -4.0, -1.0) - 15.0 * mainstage, sigma=0.8)
ftpt = add_noise(560.0 + 12.0 * smooth_step(time, -4.0, -1.0) - 10.0 * mainstage, sigma=0.8)

# Keep the older channel names as aliases in case examples still reference them.
LOX_TANK_PT = OTPT
FUEL_TANK_PT = FTPT

PNEU_SUPPLY_PT = add_noise(3000.0 - 30.0 * smooth_step(time, -4.0, 12.0), sigma=2.5)
PURGE_PT = add_noise(80.0 + 220.0 * PURGE_CMD - 50.0 * mainstage, sigma=0.8)


# ---------------------------------------------------------------------------
# Preburner and turbine temperatures, K
# ---------------------------------------------------------------------------

pbt_base = 295.0 + mainstage * (880.0 + 18.0 * np.sin(2.0 * np.pi * 0.22 * time))
pbt_ignition_spike = 120.0 * np.exp(-((time - 0.35) / 0.18) ** 2)
pbt_base += pbt_ignition_spike * mainstage

PBTC_1 = add_noise(first_order_lag(pbt_base, tau=0.18, dt=dt), sigma=3.0, drift=0.0003)
PBTC_2 = add_noise(first_order_lag(pbt_base * 0.985 + 12.0, tau=0.22, dt=dt), sigma=3.5, drift=0.0003)
PBTC_3 = add_noise(first_order_lag(pbt_base * 1.015 - 8.0, tau=0.16, dt=dt), sigma=4.0, drift=0.0004)
PBTC_4 = add_noise(first_order_lag(pbt_base * 0.995 + 4.0, tau=0.25, dt=dt), sigma=3.2)

TITC_1 = add_noise(first_order_lag(330.0 + mainstage * 720.0, tau=0.25, dt=dt), sigma=2.8)
TITC_2 = add_noise(first_order_lag(330.0 + mainstage * 700.0, tau=0.30, dt=dt), sigma=3.0)

EGT_1 = add_noise(first_order_lag(300.0 + mainstage * 520.0, tau=0.35, dt=dt), sigma=2.5)
EGT_2 = add_noise(first_order_lag(300.0 + mainstage * 500.0, tau=0.40, dt=dt), sigma=2.5)


# ---------------------------------------------------------------------------
# Shaft speed, rpm
# ---------------------------------------------------------------------------

speed_base = mainstage * (28500.0 * throttle)
speed_base += mainstage * 850.0 * np.exp(-((time - 0.65) / 0.22) ** 2)
speed_base = first_order_lag(speed_base, tau=0.16, dt=dt)

SHAFT_RPM_1 = add_noise(speed_base, sigma=22.0, drift=0.004)
SHAFT_RPM_2 = add_noise(speed_base * 1.002 - 35.0, sigma=28.0, drift=0.004)
SHAFT_RPM_3 = add_noise(first_order_lag(speed_base, tau=0.05, dt=dt) * 0.998 + 40.0, sigma=35.0)


# ---------------------------------------------------------------------------
# Flow meter traces
# ---------------------------------------------------------------------------

OX_MDOT = mainstage * (3.25 * throttle)
FUEL_MDOT = mainstage * (1.52 * throttle)

# Startup overshoots and shutdown undershoot.
OX_MDOT += mainstage * 0.20 * np.exp(-((time - 0.45) / 0.18) ** 2)
FUEL_MDOT += mainstage * 0.08 * np.exp(-((time - 0.55) / 0.20) ** 2)

OX_MDOT = add_noise(first_order_lag(OX_MDOT, tau=0.08, dt=dt), sigma=0.012)
FUEL_MDOT = add_noise(first_order_lag(FUEL_MDOT, tau=0.08, dt=dt), sigma=0.008)

MR = np.divide(OX_MDOT, FUEL_MDOT, out=np.full_like(OX_MDOT, np.nan), where=FUEL_MDOT > 0.05)
MR_NOISY = add_noise(MR, sigma=0.015)
MR_NOISY[~np.isfinite(MR_NOISY)] = 0.0


# ---------------------------------------------------------------------------
# Line temperatures and structural temperatures
# ---------------------------------------------------------------------------

LOX_LINE_TC_1 = add_noise(first_order_lag(295.0 - 205.0 * smooth_step(time, -3.8, -0.4) + 8.0 * mainstage, tau=0.35, dt=dt), sigma=0.8)
LOX_LINE_TC_2 = add_noise(first_order_lag(295.0 - 198.0 * smooth_step(time, -3.2, -0.2) + 6.0 * mainstage, tau=0.45, dt=dt), sigma=0.8)
FUEL_LINE_TC_1 = add_noise(first_order_lag(295.0 + 18.0 * mainstage, tau=0.70, dt=dt), sigma=0.5)
FUEL_LINE_TC_2 = add_noise(first_order_lag(295.0 + 22.0 * mainstage, tau=0.85, dt=dt), sigma=0.5)

CHAMBER_WALL_TC_1 = add_noise(first_order_lag(295.0 + 175.0 * mainstage, tau=1.10, dt=dt), sigma=0.7)
CHAMBER_WALL_TC_2 = add_noise(first_order_lag(295.0 + 145.0 * mainstage, tau=1.25, dt=dt), sigma=0.7)
NOZZLE_WALL_TC_1 = add_noise(first_order_lag(295.0 + 115.0 * mainstage, tau=1.40, dt=dt), sigma=0.7)
NOZZLE_WALL_TC_2 = add_noise(first_order_lag(295.0 + 95.0 * mainstage, tau=1.50, dt=dt), sigma=0.7)


# ---------------------------------------------------------------------------
# Vibration and accelerometer traces
# ---------------------------------------------------------------------------

VIBE_X = mainstage * (
    0.25 * np.sin(2.0 * np.pi * 180.0 * time)
    + 0.12 * np.sin(2.0 * np.pi * 360.0 * time + 0.4)
)
VIBE_Y = mainstage * (
    0.20 * np.sin(2.0 * np.pi * 175.0 * time + 0.2)
    + 0.10 * np.sin(2.0 * np.pi * 350.0 * time + 1.1)
)
VIBE_Z = mainstage * (
    0.30 * np.sin(2.0 * np.pi * 190.0 * time + 0.7)
    + 0.15 * np.sin(2.0 * np.pi * 380.0 * time + 0.3)
)

VIBE_X = add_noise(VIBE_X, sigma=0.035, spike_probability=0.0005, spike_sigma=1.0)
VIBE_Y = add_noise(VIBE_Y, sigma=0.035, spike_probability=0.0005, spike_sigma=1.0)
VIBE_Z = add_noise(VIBE_Z, sigma=0.040, spike_probability=0.0005, spike_sigma=1.2)


# ---------------------------------------------------------------------------
# Add a few intentionally messy sensor channels
# ---------------------------------------------------------------------------

# Sensor with mild dropout around startup.
PCMC_DROPOUT = PCMC_1.copy()
dropout_mask = (time > 0.15) & (time < 0.25)
PCMC_DROPOUT[dropout_mask] = np.nan

# Sensor with a stuck value after shutdown.
PCMC_STUCK = PCMC_2.copy()
stuck_index = np.searchsorted(time, 12.8)
PCMC_STUCK[stuck_index:] = PCMC_STUCK[stuck_index]

# High-noise duplicate pressure sensor.
PCMC_NOISY = add_noise(pc_nominal, sigma=4.5, spike_probability=0.001, spike_sigma=25.0)


# ---------------------------------------------------------------------------
# Redline / blueline style traces, still just normal traces
# ---------------------------------------------------------------------------

PCMC_REDLINE = np.full_like(time, 400.0)
PCMC_YELLOWLINE = np.full_like(time, 350.0)
PCMC_BLUELINE = np.full_like(time, 100.0)

PBTC_REDLINE = np.full_like(time, 1300.0)
SHAFT_RPM_REDLINE = np.full_like(time, 32000.0)


# ---------------------------------------------------------------------------
# Write file
# ---------------------------------------------------------------------------

channels = {
    # Commands
    "PURGE_CMD": (PURGE_CMD, "fraction", "Purge command"),
    "IGN_CMD": (IGN_CMD, "fraction", "Igniter command"),
    "MOV_CMD": (MOV_CMD, "fraction", "Main oxidizer valve command"),
    "MFV_CMD": (MFV_CMD, "fraction", "Main fuel valve command"),
    "GG_MOV_CMD": (GG_MOV_CMD, "fraction", "Gas-generator oxidizer valve command"),
    "GG_MFV_CMD": (GG_MFV_CMD, "fraction", "Gas-generator fuel valve command"),
    "VENT_CMD": (VENT_CMD, "fraction", "Vent command"),

    # Chamber pressure
    "PCMC_1": (PCMC_1, "psia", "Chamber pressure 1"),
    "PCMC_2": (PCMC_2, "psia", "Chamber pressure 2"),
    "PCMC_3": (PCMC_3, "psia", "Chamber pressure 3"),
    "PCMC_4": (PCMC_4, "psia", "Chamber pressure 4"),
    "PCMC_DROPOUT": (PCMC_DROPOUT, "psia", "Chamber pressure with startup dropout"),
    "PCMC_STUCK": (PCMC_STUCK, "psia", "Chamber pressure with stuck post-shutdown value"),
    "PCMC_NOISY": (PCMC_NOISY, "psia", "High-noise chamber pressure"),

    # Pressure
    "OIPT": (OIPT, "psia", "Oxidizer injector pressure"),
    "FIPT": (FIPT, "psia", "Fuel injector pressure"),
    "OX_MAN_1": (OX_MAN_1, "psia", "Oxidizer manifold pressure 1"),
    "OX_MAN_2": (OX_MAN_2, "psia", "Oxidizer manifold pressure 2"),
    "FUEL_MAN_1": (FUEL_MAN_1, "psia", "Fuel manifold pressure 1"),
    "FUEL_MAN_2": (FUEL_MAN_2, "psia", "Fuel manifold pressure 2"),
    # Tank pressures
    "OTPT": (OTPT, "psia", "Oxidizer tank pressure"),
    "FTPT": (FTPT, "psia", "Fuel tank pressure"),
    "ftpt": (ftpt, "psia", "Water tank pressure"),

    # Backward-compatible aliases
    "LOX_TANK_PT": (LOX_TANK_PT, "psia", "LOX tank pressure"),
    "FUEL_TANK_PT": (FUEL_TANK_PT, "psia", "Fuel tank pressure"),

    "PNEU_SUPPLY_PT": (PNEU_SUPPLY_PT, "psia", "Pneumatic supply pressure"),
    "PURGE_PT": (PURGE_PT, "psia", "Purge pressure"),

    # Temperatures
    "PBTC_1": (PBTC_1, "K", "Preburner temperature 1"),
    "PBTC_2": (PBTC_2, "K", "Preburner temperature 2"),
    "PBTC_3": (PBTC_3, "K", "Preburner temperature 3"),
    "PBTC_4": (PBTC_4, "K", "Preburner temperature 4"),
    "TITC_1": (TITC_1, "K", "Turbine inlet temperature 1"),
    "TITC_2": (TITC_2, "K", "Turbine inlet temperature 2"),
    "EGT_1": (EGT_1, "K", "Exhaust gas temperature 1"),
    "EGT_2": (EGT_2, "K", "Exhaust gas temperature 2"),
    "LOX_LINE_TC_1": (LOX_LINE_TC_1, "K", "LOX line temperature 1"),
    "LOX_LINE_TC_2": (LOX_LINE_TC_2, "K", "LOX line temperature 2"),
    "FUEL_LINE_TC_1": (FUEL_LINE_TC_1, "K", "Fuel line temperature 1"),
    "FUEL_LINE_TC_2": (FUEL_LINE_TC_2, "K", "Fuel line temperature 2"),
    "CHAMBER_WALL_TC_1": (CHAMBER_WALL_TC_1, "K", "Chamber wall temperature 1"),
    "CHAMBER_WALL_TC_2": (CHAMBER_WALL_TC_2, "K", "Chamber wall temperature 2"),
    "NOZZLE_WALL_TC_1": (NOZZLE_WALL_TC_1, "K", "Nozzle wall temperature 1"),
    "NOZZLE_WALL_TC_2": (NOZZLE_WALL_TC_2, "K", "Nozzle wall temperature 2"),

    # Speeds and flows
    "SHAFT_RPM_1": (SHAFT_RPM_1, "rpm", "Turbopump shaft speed 1"),
    "SHAFT_RPM_2": (SHAFT_RPM_2, "rpm", "Turbopump shaft speed 2"),
    "SHAFT_RPM_3": (SHAFT_RPM_3, "rpm", "Turbopump shaft speed 3"),
    "OX_MDOT": (OX_MDOT, "kg/s", "Oxidizer mass flow"),
    "FUEL_MDOT": (FUEL_MDOT, "kg/s", "Fuel mass flow"),
    "MR": (MR_NOISY, "", "Mixture ratio"),

    # Vibration
    "VIBE_X": (VIBE_X, "g", "Vibration X"),
    "VIBE_Y": (VIBE_Y, "g", "Vibration Y"),
    "VIBE_Z": (VIBE_Z, "g", "Vibration Z"),

    # Limit traces
    "PCMC_REDLINE": (PCMC_REDLINE, "psia", "Chamber pressure redline"),
    "PCMC_YELLOWLINE": (PCMC_YELLOWLINE, "psia", "Chamber pressure warning line"),
    "PCMC_BLUELINE": (PCMC_BLUELINE, "psia", "Chamber pressure lower reference line"),
    "PBTC_REDLINE": (PBTC_REDLINE, "K", "Preburner temperature redline"),
    "SHAFT_RPM_REDLINE": (SHAFT_RPM_REDLINE, "rpm", "Shaft speed redline"),
}

with h5py.File(filename, "w") as h5:
    h5.attrs["description"] = "Synthetic rocket engine hotfire sensor traces"
    h5.attrs["sample_rate_hz"] = 1.0 / dt
    h5.attrs["time_units"] = "s"

    write_channel(h5, "time", time, units="s", description="Time")

    for name, (data, units, description) in channels.items():
        write_channel(h5, name, data, units=units, description=description)

print(f"Wrote {filename.resolve()}")
print(f"Samples: {len(time)}")
print(f"Channels: {len(channels) + 1}")
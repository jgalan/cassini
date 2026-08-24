#!/usr/bin/env python3


"""
Cassini 2002 solar conjunction geometry

Downloads barycentric vectors from JPL Horizons and computes:

    - t1: Emission time
    - t2: Closest approach  (up-link)
    - t3: Cassini reception time
    - t4: Closest approach (down-link)
    - t5: Reception time
    - Shapiro time delay (Logarithmic expression)
    - Shapiro time delay (Expanded expression)
    - Derivative of each version 
    - Impact parameter evolution
    - Impact parameter rate evolution
    - Second derivative (Impact on residuals)
"""

import numpy as np
import pandas as pd

from astroquery.jplhorizons import Horizons
from astropy.time import Time

from datetime import datetime, timedelta

from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar


import os

# Physical constants (SI)

AU_KM = 149597870.7
RSUN_KM = 695700.0

GM_SUN = 1.32712440018e20      # m^3/s^2
C = 299792458.0                # m/s
AU = 149597870700.0            # m

m = 2.0 * GM_SUN / C**3        # seconds

# ---------------------------------------------------------

def get_vectors(target):

    epochs = {
        "start": "2002-06-05",
        "stop": "2002-07-08",
        "step": "10m",
    }

    obj = Horizons(
        id=target,
        location="@0", #barycentric
        epochs=epochs,
    )

    vec = obj.vectors()

    return vec

# ---------------------------------------------------------

print("Downloading Earth vectors...")

earth = get_vectors("399")

print("Downloading Cassini vectors...")

cassini = get_vectors("-82")

print("Downloading Sun vectors...")
sun = get_vectors("10")

rows = []

for e, c, s in zip(earth, cassini, sun):

    date = e["datetime_str"]

    rE = np.array([
        e["x"], e["y"], e["z"]
    ], dtype=float)

    rC = np.array([
        c["x"], c["y"], c["z"]
    ], dtype=float)

    rS = np.array([
        s["x"], s["y"], s["z"]
    ], dtype=float)

    # Earth -> Cassini
    rEC_vec = rC - rE

    # Heliocentric vectors
    rE_Sun = rE - rS
    rC_Sun = rC - rS

    # Distances
    rEC_AU = np.linalg.norm(rEC_vec)
    rE_Sun_AU = np.linalg.norm(rE_Sun)
    rC_Sun_AU = np.linalg.norm(rC_Sun)

    rows.append({
        "date": date,

        # Barycentric coordinates
        "xE": rE[0],
        "yE": rE[1],
        "zE": rE[2],

        "xC": rC[0],
        "yC": rC[1],
        "zC": rC[2],

        "xS": rS[0],
        "yS": rS[1],
        "zS": rS[2],

        # Distances
        "Earth-Cassini (AU)": rEC_AU,
        "Earth-Sun (AU)": rE_Sun_AU,
        "Cassini-Sun (AU)": rC_Sun_AU,
    })

df = pd.DataFrame(rows)

# Time grid in seconds, relative to first epoch
t = np.array([
    datetime.strptime(
        s,
        "A.D. %Y-%b-%d %H:%M:%S.%f"
    ).timestamp()
    for s in df["date"]
])

t = t - t[0]

rE_vec = df[["xE", "yE", "zE"]].to_numpy()
rC_vec = df[["xC", "yC", "zC"]].to_numpy()
rS_vec = df[["xS", "yS", "zS"]].to_numpy()

interp_E = interp1d(
    t,
    rE_vec,
    axis=0,
    kind="cubic",
    bounds_error=True
)

interp_C = interp1d(
    t,
    rC_vec,
    axis=0,
    kind="cubic",
    bounds_error=True
)

interp_S = interp1d(
    t,
    rS_vec,
    axis=0,
    kind="cubic",
    bounds_error=True
)

def light_time_downlink(t5):
    """
    Given Earth reception time t5,
    solve for Cassini transmission time t3.
    """

    rE = interp_E(t5)
    rC = interp_C(t5)

    R = np.linalg.norm(rE - rC)

    t3 = t5 - R * AU / C

    for _ in range(20):

        rC = interp_C(t3)

        R = np.linalg.norm(rE - rC)

        t3_new = t5 - R * AU / C

        if abs(t3_new - t3) < 1e-6:
            break

        t3 = t3_new

    return t3

def light_time_uplink(t3):
    """
    Given Cassini reception/transmission time t3,
    solve for Earth transmission time t1.
    """

    rC = interp_C(t3)
    rE = interp_E(t3)

    R = np.linalg.norm(rC - rE)

    t1 = t3 - R * AU / C

    for _ in range(20):

        rE = interp_E(t1)

        R = np.linalg.norm(rC - rE)

        t1_new = t3 - R * AU / C

        if abs(t1_new - t1) < 1e-6:
            break

        t1 = t1_new

    return t1

def closest_approach_with_bdot(t_start, t_end, r_start, r_end, dt):
    """Return closest approach and db/dt."""

    # Central value
    _, b0, bvec0 = closest_approach(
        t_start, t_end, r_start, r_end
    )

    # Forward trajectory
    r_start_p = interp_E(t_start + dt)
    r_end_p   = interp_C(t_end + dt)

    _, bp, _ = closest_approach(
        t_start + dt,
        t_end + dt,
        r_start_p,
        r_end_p
    )

    # Backward trajectory
    r_start_m = interp_E(t_start - dt)
    r_end_m   = interp_C(t_end - dt)

    _, bm, _ = closest_approach(
        t_start - dt,
        t_end - dt,
        r_start_m,
        r_end_m
    )

    b_dot = (bp - bm) / (2 * dt)

    return b0, b_dot

def closest_approach(t_start, t_end, r_start, r_end):
    """
    Given an emission time t_start and a reception time t_end,
    we construct the straight-line trajectory and find time
    at which the sun is found at the closest distance

    Returns the time, impact parameter, and pointing vector
    """

    def distance_to_sun(tt):

        alpha = (tt - t_start) / (t_end - t_start)

        r_gamma = r_start + alpha * (r_end - r_start)

        rS = interp_S(tt)

        return np.linalg.norm(r_gamma - rS)

    result = minimize_scalar(
        distance_to_sun,
        bounds=(t_start, t_end),
        method="bounded",
        options={"xatol": 1e-6}
    )

    tt = result.x

    alpha = (tt - t_start) / (t_end - t_start)

    r_gamma = r_start + alpha * (r_end - r_start)
    rS = interp_S(tt)

    b_vec = r_gamma - rS
    b = np.linalg.norm(b_vec)

    return tt, b, b_vec

###################################
print("\nTwo-way light-time test")

for i in range(10000, len(t), 300):

    t3 = t[i]

    t2 = light_time_downlink(t3)
    t1 = light_time_uplink(t2)

    print(
        f"t3 = {t3:10.1f} s   "
        f"t2 = {t2:10.1f} s   "
        f"t1 = {t1:10.1f} s   "
        f"down = {t3-t2:8.2f} s   "
        f"up = {t2-t1:8.2f} s"
    )

###################################
print("\nDetermining t1,t3,t5 arrays")
t1_array = []
t3_array = []
t5_array = []


for i in range(len(t)):
    t5 = t[i]

    # Skipping first 3-hours
    if t5 < 10000:
        continue

    t3 = light_time_downlink(t5)
    t1 = light_time_uplink(t3)

    t1_array.append(t1)
    t3_array.append(t3)
    t5_array.append(t5)

t1_array = np.array(t1_array)
t3_array = np.array(t3_array)
t5_array = np.array(t5_array)

#######################################

###################################
print("\nDetermining t2,t4 arrays")
t2_array = []
t4_array = []

b_up_array = []
b_down_array = []

for t1, t3, t5 in zip(t1_array, t3_array, t5_array):

    rE1 = interp_E(t1)
    rC3 = interp_C(t3)
    rE5 = interp_E(t5)

    t2, b_up, _ = closest_approach(
        t1, t3,
        rE1, rC3
    )

    b_up_f, db_up_f = closest_approach_with_bdot(
        t1, t3,
        rE1, rC3,
        1
    )

    t4, b_down, _ = closest_approach(
        t3, t5,
        rC3, rE5
    )

    t2_array.append(t2)
    t4_array.append(t4)

    # AU -> solar radii
    b_up_array.append(b_up * AU_KM / RSUN_KM)
    b_down_array.append(b_down * AU_KM / RSUN_KM)

t2_array = np.array(t2_array)
t4_array = np.array(t4_array)

b_up_array = np.array(b_up_array)
b_down_array = np.array(b_down_array)

def shapiro_uplink(t1, t2, t3):

    rE1 = interp_E(t1)
    rC3 = interp_C(t3)
    rS2 = interp_S(t2)

    # We define distances respect to the sun position at the closest approach (at t2)
    R1 = np.linalg.norm(rE1 - rS2)
    R3 = np.linalg.norm(rC3 - rS2)
    R13 = np.linalg.norm(rC3 - rE1)

    return m * np.log(
        (R1 + R3 + R13) /
        (R1 + R3 - R13)
    )


def shapiro_downlink(t3, t4, t5):

    rC3 = interp_C(t3)
    rE5 = interp_E(t5)
    rS4 = interp_S(t4)

    # We define distances respect to the sun position at the closest approach (at t4)
    R3 = np.linalg.norm(rC3 - rS4)
    R5 = np.linalg.norm(rE5 - rS4)
    R35 = np.linalg.norm(rE5 - rC3)

    return m * np.log(
        (R3 + R5 + R35) /
        (R3 + R5 - R35)
    )

dt_up_array = np.empty(len(t5_array))
dt_down_array = np.empty(len(t5_array))

for i, (t1, t2, t3, t4, t5) in enumerate(
        zip(t1_array, t2_array, t3_array, t4_array, t5_array)):

    dt_up_array[i] = shapiro_uplink(t1, t2, t3)
    dt_down_array[i] = shapiro_downlink(t3, t4, t5)

dt_shapiro_array = dt_up_array + dt_down_array

y_shapiro = np.gradient(
    dt_shapiro_array,
    t5_array
)

i = np.argmax(np.abs(y_shapiro))

y_up_shapiro = np.gradient(
    dt_up_array,
    t5_array
)

i = np.argmax(np.abs(y_shapiro))

print("Maximum |y_shapiro|:")
print("y =", y_shapiro[i])
print("y_up =", y_up_shapiro[i])
print("t1 =", t1_array[i])
print("t2 =", t2_array[i])
print("t3 =", t3_array[i])
print("t4 =", t4_array[i])
print("t5 =", t5_array[i])

# Maximum conjunction = minimum impact parameter
i_conj = np.argmin(b_down_array)
t_conj = t4_array[i_conj]

# Time relative to conjunction, in days
time_days = (np.array(t4_array) - t_conj) / 86400.0

db_up_dt = np.gradient(b_up_array, t2_array) * RSUN_KM       # km/s
db_down_dt = np.gradient(b_down_array, t4_array) * RSUN_KM  # km/s


# Save tab-separated file
data = np.column_stack((time_days, y_up_shapiro, b_down_array, b_up_array, db_up_dt, db_down_dt))

np.savetxt(
    "params_vs_time.dat",
    data,
    delimiter="\t",
    header="#time_days\ty_shapiro\tb_up\tb_down\tdb_up_dt\tdb_down_dt",
    comments=""
)


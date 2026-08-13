#!/usr/bin/env python3


"""
Cassini 2002 solar conjunction geometry

Downloads heliocentric vectors from JPL Horizons and computes:

    Date
    Sun-Earth distance
    Sun-Cassini distance
    Earth-Cassini distance
    Impact parameter

Results are written to geometry.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astroquery.jplhorizons import Horizons
from astropy.time import Time
from datetime import datetime


# Physical constants (SI)

AU_KM = 149597870.7
RSUN_KM = 695700.0

GM_SUN = 1.32712440018e20      # m^3/s^2
C = 299792458.0                # m/s
AU = 149597870700.0            # m

K = 2.0 * GM_SUN / C**3        # seconds


# ---------------------------------------------------------

def get_vectors(target):

    epochs = {
        "start": "2002-06-06",
        "stop": "2002-07-07",
        "step": "1m",
    }

    obj = Horizons(
        id=target,
        location="@10",      # Sun
        epochs=epochs,
    )

    vec = obj.vectors()

    return vec


# ---------------------------------------------------------

print("Downloading Earth vectors...")

earth = get_vectors("399")

print("Downloading Cassini vectors...")

cassini = get_vectors("-82")

rows = []

for e, c in zip(earth, cassini):

    date = e["datetime_str"]

    rE = np.array([e["x"], e["y"], e["z"]], dtype=float)
    rC = np.array([c["x"], c["y"], c["z"]], dtype=float)

    rEC_vec = rC - rE

    rE_AU = np.linalg.norm(rE)
    rC_AU = np.linalg.norm(rC)
    rEC_AU = np.linalg.norm(rEC_vec)

    b_AU = np.linalg.norm(np.cross(rE, rC)) / rEC_AU

    rows.append({
        "date": date,
        "Sun-Earth (AU)": rE_AU,
        "Sun-Cassini (AU)": rC_AU,
        "Earth-Cassini (AU)": rEC_AU,
        "Impact parameter (AU)": b_AU,
        "Impact parameter (Rsun)": b_AU * AU_KM / RSUN_KM,
    })

df = pd.DataFrame(rows)

####
t = np.array([
    datetime.strptime(
        s,
        "A.D. %Y-%b-%d %H:%M:%S.%f"
    ).timestamp()
    for s in df["date"]
])

# Distancias en metros
rE = df["Sun-Earth (AU)"].to_numpy() * AU
rC = df["Sun-Cassini (AU)"].to_numpy() * AU
R  = df["Earth-Cassini (AU)"].to_numpy() * AU

b  = df["Impact parameter (AU)"] * AU

# Distancia desde la Tierra al punto de máxima aproximación
df["LE (AU)"] = np.sqrt(rE**2 - b**2)

# Distancia desde Cassini al punto de máxima aproximación
df["LC (AU)"] = np.sqrt(rC**2 - b**2)

# Cocientes adimensionales
df["LE/rE"] = df["LE (AU)"] / rE
df["LC/rC"] = df["LC (AU)"] / rC

# Suma
df["LE/rE + LC/rC"] = df["LE/rE"] + df["LC/rC"] - 2

lE = df["LE/rE"].to_numpy() 
lC = df["LC/rC"].to_numpy() 

lsum = GM_SUN / C**3 * ( lE + lC )

# Shapiro
df["Shapiro (s)"] = (
    2 * GM_SUN / C**3
    * np.log((rE + rC + R) / (rE + rC - R))
)
df["Shapiro (us)"] = 1e6 * df["Shapiro (s)"]

delay = df["Shapiro (s)"].to_numpy()

# Shapiro JG
df["Shapiro JG (s)"] = (
    2 * GM_SUN / C**3
    * np.log((rE + rC + R) / (rE + rC - R)) + GM_SUN / C**3 * (lE + lC)
)
df["Shapiro JG (us)"] = 1e6 * df["Shapiro JG (s)"]

delayJG = df["Shapiro JG (s)"].to_numpy()

y = np.gradient(delay, t)
yJG = np.gradient(delayJG, t)

df["y"] = y
df["y x1e13"] = 1e13 * y
df["yJG"] = yJG
df["yJG x1e13"] = 1e13 * yJG

imax = np.argmax(np.abs(y))

print(df.loc[imax, [
    "date",
    "Impact parameter (Rsun)"
]])

print("Maximum |y| =", np.abs(y[imax]))

imax = df["Shapiro (us)"].idxmax()

print("\nMaximum Shapiro delay")
print(df.loc[imax, [
    "date",
    "Impact parameter (Rsun)",
    "Shapiro (us)"
]])

rE_AU = df["Sun-Earth (AU)"]
rC_AU = df["Sun-Cassini (AU)"]
R_AU  = df["Earth-Cassini (AU)"]
b_AU  = df["Impact parameter (AU)"]

df["LE (AU)"] = np.sqrt(rE_AU**2 - b_AU**2)
df["LC (AU)"] = np.sqrt(rC_AU**2 - b_AU**2)

df["LE/rE"] = df["LE (AU)"]/rE_AU
df["LC/rC"] = df["LC (AU)"]/rC_AU

df["Shapiro approx (s)"] = (
    2 * GM_SUN / C**3
    * np.log(4 * rE_AU * rC_AU / b_AU**2)
)

df["Difference (ns)"] = (
    1e9
    * (df["Shapiro (s)"] - df["Shapiro approx (s)"])
)

print(df[[
    "date",
    "Shapiro (us)",
    "Difference (ns)"
]].round(6))

print(
    df[
        [
            "date",
            "LE (AU)",
            "LC (AU)",
            "LE/rE",
            "LC/rC",
            "LE/rE + LC/rC",
        ]
    ].round(8)
)

imin = df["Impact parameter (Rsun)"].idxmin()

print("\nClosest approach to the Sun")
print(df.loc[imin])

#print(df)

df.to_csv("geometry.csv", index=False)

print()
print("Saved geometry.csv")


t_datetime = [
    datetime.strptime(s, "A.D. %Y-%b-%d %H:%M:%S.%f")
    for s in df["date"]
]

dt = (t_datetime[1] - t_datetime[0]).total_seconds()

imin = df["Impact parameter (AU)"].idxmin()
t0 = imin * dt      # dt = 60 s
t_rel = (np.arange(len(df)) * dt - t0) / 86400.0

t_sec = t_rel * 86400.0
dt = t_sec

# Time derivatives (AU/s)
drE = np.gradient(rE, dt)
drC = np.gradient(rC, dt)
dR  = np.gradient(R, dt)
dlsum = np.gradient(lsum, dt)
d2lsum = np.gradient(dlsum, dt) * 8 * 3600

# Exact analytical derivative
y_exact = K * (
    (drE + drC + dR)/(rE + rC + R)
    -
    (drE + drC - dR)/(rE + rC - R)
)

# Numerical derivative
y_num = np.gradient(delay, dt)

# ------------------------------------------------------------
# Comparison
# ------------------------------------------------------------

plt.figure(figsize=(9,5))

plt.plot(t_rel,1e13*y_num,label="Numerical derivative")
plt.plot(t_rel,1e13*y_exact,"--",label="Analytical derivative")

plt.xlabel("Days from conjunction")
plt.ylabel(r"$10^{13}\,\Delta\nu/\nu$")
plt.grid(True)
plt.legend()

# ------------------------------------------------------------
# Difference
# ------------------------------------------------------------

difference = y_num - y_exact

print()
print("Maximum absolute difference")
print(np.max(np.abs(difference)))

print()
print("Relative difference")
print(np.max(np.abs(difference))/np.max(np.abs(y_num)))

# Numerical derivative
y_num = np.gradient(delay, dt)

plt.figure(figsize=(10,5))

plt.title("Cassini Doppler observables (y and yJG)")
plt.plot( t_rel, 1e13*y, "b-", lw=2, label="RG")
plt.plot(t_rel, 1e13*yJG, "r--",  lw=2, label="JG")

plt.xlabel("Days")
plt.ylabel(r"$10^{13}\,\Delta\nu/\nu$")
plt.grid(True)
plt.legend()

plt.figure(figsize=(10,6))

plt.title("Cassini Doppler observable (Difference y-yJG)")
plt.plot(t_rel, 1e13*(yJG - y), lw=2)

plt.xlabel("Days from conjunction")
plt.ylabel(r"$10^{13}\,\Delta(\Delta\nu/\nu)$")
plt.grid(True)


plt.figure(figsize=(10,5))
plt.title("lE and lC)")

plt.plot(t_rel, lE, label="lE")
plt.plot(t_rel, lC, label="lC")

plt.title("Line-of-sight distances")
plt.xlabel("Days from conjunction")
plt.ylabel("Distance (AU)")
plt.grid(True)
plt.legend()


plt.figure(figsize=(10,4))
plt.plot(t_rel, df["Impact parameter (Rsun)"])
plt.xlabel("Days from conjunction")
plt.ylabel("Impact parameter ($R_\\odot$)")
plt.grid()

plt.figure(figsize=(10,4))
plt.plot(t_rel, 1e6*delay)
plt.xlabel("Days from conjunction")
plt.ylabel("Shapiro delay (μs)")
plt.grid()

mask = np.abs(t_rel) < 2.0

plt.figure(figsize=(10,4))
plt.plot(t_rel[mask],1e13*y[mask])
plt.grid()
plt.xlabel("Days from conjunction")
plt.ylabel(r"$10^{13}\Delta\nu/\nu$")

plt.figure(figsize=(10,6))

#plt.plot(t_rel, dlsum, lw=2, label="D")
plt.plot(t_rel, d2lsum, lw=2, label="D2")

plt.title(r"Derivative of $l_E+l_C$")
plt.xlabel("Days from conjunction")
plt.ylabel(r"$d(l_E+l_C)/dt$")
plt.grid(True)
plt.legend()

plt.show()

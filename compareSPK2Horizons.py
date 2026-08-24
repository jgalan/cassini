import numpy as np
from astroquery.jplhorizons import Horizons
import spiceypy as spice

AU_KM = 149597870.7

# =========================================================
# SPICE kernels
# =========================================================

SPK = "data/c32easc2002_160_2002_186.bsp"
LSK = "data/lsk_981207.tls"

spice.furnsh(LSK)
spice.furnsh(SPK)

# =========================================================
# Horizons
# =========================================================

def get_vectors(target):

    epochs = {
        "start": "2002-06-05",
        "stop": "2002-07-08",
        "step": "1m",
    }

    obj = Horizons(
        id=target,
        location="@10",
        epochs=epochs,
    )

    vec = obj.vectors()

    print("\n=== HORIZONS INFORMATION ===")
    print("Columns:")
    print(vec.columns)

    print("\nMetadata:")
    print(vec.meta)

    print("\nFirst row:")
    print(vec[0])

    return obj.vectors()


print("Downloading Cassini vectors...")
cassini = get_vectors("-82")

# =========================================================
# Target epoch
# =========================================================

target_et = spice.str2et(
    "2002 JUN 20 00:00:00 TDB"
)

target_jd = spice.unitim(
    target_et,
    "ET",
    "JDTDB"
)

# Find nearest Horizons row
jd = np.array([
    float(row["datetime_jd"])
    for row in cassini
])

i = np.argmin(np.abs(jd - target_jd))

h_row = cassini[i]

h_cassini = np.array([
    float(h_row["x"]),
    float(h_row["y"]),
    float(h_row["z"])
]) * AU_KM

print("\nHorizons epoch:")
print(h_row["datetime_str"])

print(
    "Epoch difference [s]:",
    (jd[i] - target_jd) * 86400
)

# =========================================================
# CAS-NAV
# =========================================================

state, lt = spice.spkezr(
    "-82",
    target_et,
    "J2000",
    "NONE",
    "SUN"
)

spk_cassini = np.array(state[:3])

# ---------------------------------------------------------
# Transform Horizons vector from ECLIPJ2000 to J2000
# ---------------------------------------------------------

rot = spice.pxform(
    "ECLIPJ2000",
    "J2000",
    target_et
)

h_j2000 = rot @ h_cassini

print("\nHorizons original [km]:")
print(h_cassini)

print("\nHorizons transformed to J2000 [km]:")
print(h_j2000)

print("\nCAS-NAV SPK [km]:")
print(spk_cassini)

difference = h_j2000 - spk_cassini

print("\nDifference [km]:")
print(difference)

print("\n|Difference| [km]:")
print(np.linalg.norm(difference))

# =========================================================
# Comparison
# =========================================================

difference = spk_cassini - h_cassini

print("\n" + "=" * 70)
print("CASSINI POSITION COMPARISON")
print("=" * 70)

print("\nCAS-NAV SPK [km]:")
print(spk_cassini)

print("\nHorizons [km]:")
print(h_cassini)

print("\nDifference [km]:")
print(difference)

print("\n|r|:")
print("CAS-NAV  :", np.linalg.norm(spk_cassini))
print("Horizons :", np.linalg.norm(h_cassini))

print("\n|Delta r|:")
print(np.linalg.norm(difference))

print("\nRelative difference:")
print(
    np.linalg.norm(difference)
    / np.linalg.norm(spk_cassini)
)

spice.kclear()

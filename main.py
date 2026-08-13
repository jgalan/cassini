import requests
import pandas as pd
from io import StringIO

BASE = "https://ssd.jpl.nasa.gov/api/horizons.api"

START = "2002-06-06"
STOP  = "2002-07-07"
STEP  = "1 d"


def horizons_vectors(command, center):
    """
    Devuelve una tabla de vectores desde 'center' hasta 'command'
    """

    params = {
        "format": "text",
        "COMMAND": command,
        "CENTER": center,
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "VECTORS",
        "START_TIME": START,
        "STOP_TIME": STOP,
        "STEP_SIZE": STEP,
        "OUT_UNITS": "AU-D",
        "REF_PLANE": "ECLIPTIC",
        "VEC_TABLE": "2"
    }

    r = requests.get(BASE, params=params)
    txt = r.text

    start = txt.find("$$SOE")
    end   = txt.find("$$EOE")

    if start < 0:
        raise RuntimeError(txt)

    data = txt[start+5:end]

    rows = []

    for line in data.splitlines():

        if "A.D." in line:

            date = line.strip()

        elif line.strip().startswith("X"):

            pieces = line.replace("=", " ").split()

            x = float(pieces[1])
            y = float(pieces[3])
            z = float(pieces[5])

            r = (x*x+y*y+z*z)**0.5

            rows.append([date, r])

    return pd.DataFrame(rows, columns=["date","distance_AU"])


#------------------------

print("Earth from Sun")
earth = horizons_vectors("399", "@10")

print("Cassini from Sun")
cassini = horizons_vectors("-82", "@10")

print("Cassini from Earth")
earth_cassini = horizons_vectors("-82", "399")


table = pd.DataFrame({
    "Date": earth.date,
    "Sun-Earth (AU)": earth.distance_AU,
    "Sun-Cassini (AU)": cassini.distance_AU,
    "Earth-Cassini (AU)": earth_cassini.distance_AU
})

table.to_csv("cassini_geometry.csv", index=False)

print(table)

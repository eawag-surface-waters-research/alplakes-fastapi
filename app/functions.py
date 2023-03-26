import os
import json
import shutil
import requests
import numpy as np
from fastapi import HTTPException
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta, SU


def convert_to_unit(time, units):
    if units == "seconds since 2008-03-01 00:00:00":
        return (time.replace(tzinfo=timezone.utc) - datetime(2008, 3, 1).replace(tzinfo=timezone.utc)).total_seconds()
    elif units == "seconds since 1970-01-01 00:00:00":
        return time.timestamp()
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies unable to read NetCDF with time unit: {}".format(units))


def convert_from_unit(time, units):
    if units == "seconds since 2008-03-01 00:00:00":
        return datetime.utcfromtimestamp(time + (datetime(2008, 3, 1).replace(tzinfo=timezone.utc) - datetime(1970, 1, 1).replace(tzinfo=timezone.utc)).total_seconds())
    elif units == "seconds since 1970-01-01 00:00:00":
        return datetime.utcfromtimestamp(time)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies unable to read NetCDF with time unit: {}".format(units))


def get_closest_index(value, array):
    array = np.asarray(array)
    sorted_array = np.sort(array)
    if len(array) == 0:
        raise ValueError("Array must be longer than len(0) to find index of value")
    elif len(array) == 1:
        return 0
    if value > (2 * sorted_array[-1] - sorted_array[-2]):
        raise HTTPException(status_code=400,
                            detail="Value {} greater than max available ({})".format(value, sorted_array[-1]))
    elif value < (2 * sorted_array[0] - sorted_array[-1]):
        raise HTTPException(status_code=400,
                            detail="Value {} less than min available ({})".format(value, sorted_array[0]))
    return (np.abs(array - value)).argmin()


def daterange(start_date, end_date, days=1):
    for n in range(int((end_date - start_date).days / days)):
        yield start_date + timedelta(n * days)


def array_to_list(arr):
    dims = arr.shape
    if len(dims) == 1:
        return list(arr)
    elif len(dims) == 2:
        out = []
        for i in range(dims[0]):
            out.append(list(arr[i]))
        return out
    elif len(dims) == 3:
        out = []
        for i in range(dims[0]):
            out2 = []
            for j in range(dims[1]):
                out2.append(list(arr[i, j]))
            out.append(out2)


def download_file(url, local):
    """Stream remote file and replace existing."""
    exists = False
    if os.path.isfile(local):
        exists = True
        old = local
        local = local.replace(".nc", "_temp.nc")
        if os.path.isfile(local):
            os.remove(local)
    elif not os.path.exists(os.path.dirname(local)):
        os.makedirs(os.path.dirname(local))
    with requests.get(url, stream=True) as r:
        with open(local, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    if exists:
        os.remove(old)
        shutil.move(local, old)


def filter_coordinate(x):
    x = np.asarray(x).astype(np.float64)
    x[x == 0.] = np.nan
    return json.dumps(np.around(x, decimals=2).tolist())


def filter_parameter(x):
    x = np.asarray(x).astype(np.float64)
    x[x == -999.0] = np.nan
    return json.dumps(np.around(x, decimals=2).tolist())


def rotate_velocity(u, v, alpha):
    u = np.asarray(u).astype(np.float64)
    v = np.asarray(v).astype(np.float64)
    alpha = np.asarray(alpha).astype(np.float64)

    u[u == -999.0] = np.nan
    v[v == -999.0] = np.nan
    alpha[alpha == 0.0] = np.nan

    alpha = np.radians(alpha)
    u_n = u * np.cos(alpha) - v * np.sin(alpha)
    v_e = v * np.cos(alpha) + u * np.sin(alpha)

    return json.dumps(np.around(u_n, decimals=5).tolist()), json.dumps(np.around(v_e, decimals=5).tolist())


def alplakes_coordinates(x, y):
    x = np.asarray(x).astype(np.float64)
    y = np.asarray(y).astype(np.float64)
    x[x == 0.] = np.nan
    y[y == 0.] = np.nan
    lat, lng = ch1903_to_latlng(x, y)
    return np.concatenate((lat, lng), axis=1)


def alplakes_temperature(x):
    x = np.asarray(x).astype(np.float64)
    x[x == -999.0] = np.nan
    return x


def alplakes_velocity(u, v, alpha):
    u = np.asarray(u).astype(np.float64)
    v = np.asarray(v).astype(np.float64)
    alpha = np.asarray(alpha).astype(np.float64)

    u[u == -999.0] = np.nan
    v[v == -999.0] = np.nan
    alpha[alpha == 0.0] = np.nan

    alpha = np.radians(alpha)
    u_n = u * np.cos(alpha) - v * np.sin(alpha)
    v_e = v * np.cos(alpha) + u * np.sin(alpha)

    return np.concatenate((u_n, v_e), axis=2)


def alplakes_time(t, units):
    return np.array([convert_from_unit(x, units).strftime("%Y%m%d%H%M") for x in t])


def sundays_between_dates(start, end, max_weeks=10):
    sunday_start = start + relativedelta(weekday=SU(-1))
    sunday_end = end + relativedelta(weekday=SU(-1))
    weeks = []
    current = sunday_start
    while current <= sunday_end and max_weeks > 0:
        weeks.append(current)
        current = current + timedelta(days=7)
        max_weeks = max_weeks - 1
    return weeks


def latlng_to_ch1903(lat, lng):
    lat = lat * 3600
    lng = lng * 3600
    lat_aux = (lat - 169028.66) / 10000
    lng_aux = (lng - 26782.5) / 10000
    x = 2600072.37 + 211455.93 * lng_aux - 10938.51 * lng_aux * lat_aux - 0.36 * lng_aux * lat_aux ** 2 - 44.54 * lng_aux ** 3 - 2000000
    y = 1200147.07 + 308807.95 * lat_aux + 3745.25 * lng_aux ** 2 + 76.63 * lat_aux ** 2 - 194.56 * lng_aux ** 2 * lat_aux + 119.79 * lat_aux ** 3 - 1000000
    return x, y


def ch1903_to_latlng(x, y):
    x_aux = (x - 600000) / 1000000
    y_aux = (y - 200000) / 1000000
    lat = 16.9023892 + 3.238272 * y_aux - 0.270978 * x_aux ** 2 - 0.002528 * y_aux ** 2 - 0.0447 * x_aux ** 2 * y_aux - 0.014 * y_aux ** 3
    lng = 2.6779094 + 4.728982 * x_aux + 0.791484 * x_aux * y_aux + 0.1306 * x_aux * y_aux ** 2 - 0.0436 * x_aux ** 3
    lat = (lat * 100) / 36
    lng = (lng * 100) / 36
    return lat, lng

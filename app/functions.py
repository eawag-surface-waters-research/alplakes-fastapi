import os
import json
import math
import shutil
import requests
import numpy as np
import pandas as pd
from fastapi import HTTPException
from typing import Dict, List, Union, Any
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta, SU


class VariableKeyModel1D(BaseModel):
    data: Union[List[Any], float, None]
    unit: Union[str, None] = None
    description: Union[str, None] = None

class VariableKeyModel2D(BaseModel):
    data: List[List[Any]]
    unit: Union[str, None] = None
    description: Union[str, None] = None

class TaskResponseModel(BaseModel):
    message: str
    status_code: int


def convert_to_unit(time, units):
    if units == "seconds since 2008-03-01 00:00:00":
        return (time.replace(tzinfo=timezone.utc) - datetime(2008, 3, 1).replace(tzinfo=timezone.utc)).total_seconds()
    elif units == "seconds since 1970-01-01 00:00:00":
        return time.timestamp()
    elif units == "nano":
        return time.timestamp() * 1000000000
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies unable to read NetCDF with time unit: {}".format(units))


def convert_from_unit(time, units):
    if units == "seconds since 2008-03-01 00:00:00":
        return datetime.utcfromtimestamp(time + (datetime(2008, 3, 1).replace(tzinfo=timezone.utc) - datetime(1970, 1, 1).replace(tzinfo=timezone.utc)).total_seconds())
    elif units == "seconds since 1970-01-01 00:00:00":
        return datetime.utcfromtimestamp(float(time))
    elif units == "nano":
        return datetime.utcfromtimestamp(time / 1000000000)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies unable to read NetCDF with time unit: {}".format(units))


def meteostation_variables():
    return {
        "air_pressure": {"unit": "hpa", "description": "Air pressure 2 m above ground", "agg": "mean"},
        "relative_humidity": {"unit": "%", "description": "Relative humidity 2 m above ground", "agg": "mean"},
        "vapour_pressure": {"unit": "hPa", "description": "Vapour pressure 2 m above ground", "agg": "mean"},
        "global_radiation": {"unit": "W/m²", "description": "Global radiation", "agg": "mean"},
        "air_temperature": {"unit": "°C", "description": "Air temperature 2 m above ground", "agg": "mean"},
        "precipitation": {"unit": "mm", "description": "Precipitation", "agg": "sum"},
        "wind_speed": {"unit": "m/s", "description": "Wind speed scalar", "agg": "mean"},
        "wind_direction": {"unit": "°", "description": "Wind direction", "agg": "mean"},
        "cloud_cover": {"unit": "%", "description": "Cloud cover", "agg": "mean"}
    }

def kelvin_to_celsius(x):
    return x - 273.15

def hourly_joules_to_watts(x):
    return x / 3600

def ten_minute_joules_cm_to_watts_m(x):
    return (x * 10 ** 4)/600

def negative_to_zero(x):
    return np.where(x < 0, 0, x)

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


def get_closest_location(latitude, longitude, lat_grid, lon_grid, yx=False):
    lon1, lat1, lon2, lat2 = map(np.radians, [longitude, latitude, lon_grid, lat_grid])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371000 * c
    if yx:
        y_index, x_index = np.unravel_index(np.nanargmin(distance), distance.shape)
    else:
        x_index, y_index = np.unravel_index(np.nanargmin(distance), distance.shape)
    return x_index, y_index, np.nanmin(distance)


def daterange(start_date, end_date, days=1):
    for n in range(int((end_date - start_date).days / days)):
        yield start_date + timedelta(n * days)


def monthrange(start_date, end_date, months=1):
    current_date = start_date
    while current_date < end_date:
        yield current_date
        current_date += relativedelta(months=months)


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


def filter_variable(x, decimals=3, string=False, nodata=-999.0):
    x = np.asarray(x).astype(float)
    if isinstance(nodata, list):
        for nd in nodata:
            x[x == nd] = None
    else:
        x[x == nodata] = None
    out = np.around(x, decimals=decimals)
    out = np.where(np.isnan(out), None, out)
    if string:
        return json.dumps(out.tolist())
    else:
        return out.tolist()


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

    return u_n, v_e


def alplakes_variable(x):
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
    try:
        return [convert_from_unit(x, units).replace(tzinfo=timezone.utc) for x in t]
    except:
        return convert_from_unit(t, units).replace(tzinfo=timezone.utc)


def unix_time(t, units):
    return np.array([convert_from_unit(x, units).timestamp() for x in t])


def exact_line_segments(lat1, lng1, lat2, lng2, lat_grid, lng_grid, start, grid):
    distance = haversine(lat1, lng1, lat2, lng2)
    n = math.ceil((distance * 1000) / (grid / 2))
    spacing = np.arange(n + 1) * (distance * 1000 / n) + start

    # calculate bearings between the two points
    lat1, lng1, lat2, lng2 = np.radians([lat1, lng1, lat2, lng2])
    bearings = np.arctan2(np.sin(lng2 - lng1) * np.cos(lat2),
                          np.cos(lat1) * np.sin(lat2) -
                          np.sin(lat1) * np.cos(lat2) * np.cos(lng2 - lng1))

    # create array of distances between the points
    distances = np.linspace(0, distance, n + 1)

    # calculate latitudes and longitudes of the points using vectorized calculations
    lats = np.arcsin(np.sin(lat1) * np.cos(distances / 6371) +
                     np.cos(lat1) * np.sin(distances / 6371) * np.cos(bearings))
    lngs = lng1 + np.arctan2(np.sin(bearings) * np.sin(distances / 6371) * np.cos(lat1),
                             np.cos(distances / 6371) - np.sin(lat1) * np.sin(lats))

    lats, lngs = np.degrees([lats, lngs])

    x_indexs, y_indexs, dists = [], [], []
    for i in range(n + 1):
        x_index, y_index, dist = get_closest_location(lats[i], lngs[i], lat_grid, lng_grid)
        x_indexs.append(x_index)
        y_indexs.append(y_index)
        dists.append(dist)

    df = pd.DataFrame(list(zip(x_indexs, y_indexs, dists, spacing)), columns=['xi', 'yi', 'dist', 'spacing'])
    df_temp = df.sort_values(['xi', 'yi', 'dist'])
    df_temp = df_temp.drop_duplicates(['xi', 'yi'], keep='first')

    if not df.iloc[0].equals(df_temp.iloc[0]):
        df_temp = pd.concat([df.iloc[[0]], df_temp], ignore_index=True)

    if not df.iloc[-1].equals(df_temp.iloc[-1]):
        df_temp = pd.concat([df.iloc[[-1]], df_temp], ignore_index=True)
    df = df_temp.sort_values(['spacing'])
    df["valid"] = True
    df.loc[df['dist'] > grid, 'valid'] = False

    return np.array(df["xi"]), np.array(df["yi"]), np.array(df["spacing"]), np.array(df["valid"]), distance * 1000


def line_segments(x1, y1, x2, y2, x, y, indexes, start, grid_spacing, yx=False):
    distance = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    n = math.ceil(distance / (grid_spacing / 2))
    spacing = np.arange(n + 1) * (distance / n) + start
    x_index, y_index, dists = [], [], []
    for i in range(n + 1):
        t = i / (n - 1)
        xx = x1 + t * (x2 - x1)
        yy = y1 + t * (y2 - y1)
        distances = np.full(x.shape, np.inf)
        distances[indexes] = ((x[indexes] - xx)**2 + (y[indexes] - yy)**2)**0.5
        if yx:
            y_i, x_i = np.unravel_index(np.nanargmin(distances), distances.shape)
            dists.append(distances[y_i, x_i])
        else:
            x_i, y_i = np.unravel_index(np.nanargmin(distances), distances.shape)
            dists.append(distances[x_i, y_i])
        x_index.append(x_i)
        y_index.append(y_i)

    df = pd.DataFrame(list(zip(x_index, y_index, dists, spacing)), columns=['xi', 'yi', 'dist', 'spacing'])
    df["valid"] = True
    df.loc[df['dist'] > grid_spacing, 'valid'] = False

    border_idx = false_indices_near_true(df["valid"])

    df_temp = df.sort_values(['xi', 'yi', 'dist'])
    df_temp = df_temp.drop_duplicates(['xi', 'yi'], keep='first')

    if not df.iloc[0].equals(df_temp.iloc[0]):
        df_temp = pd.concat([df.iloc[[0]], df_temp], ignore_index=True)

    if not df.iloc[-1].equals(df_temp.iloc[-1]):
        df_temp = pd.concat([df.iloc[[-1]], df_temp], ignore_index=True)

    for idx in border_idx:
        if df.loc[idx, "spacing"] not in df_temp["spacing"].values:
            df_temp = pd.concat([df.loc[[idx]], df_temp], ignore_index=True)

    df = df_temp.sort_values(['spacing'])

    return np.array(df["xi"]), np.array(df["yi"]), np.array(df["spacing"]), np.array(df["valid"]), distance


def false_indices_near_true(arr):
    arr = np.array(arr)
    false_indices = np.where(arr == False)[0]
    adjacent_true = np.roll(arr, 1) | np.roll(arr, -1)
    return false_indices[adjacent_true[false_indices]]


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the Earth's surface given their latitudes and longitudes.
    """
    R = 6371.0  # Earth's radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


def average_grid_spacing(latitudes, longitudes):
    """
    Calculate the average grid spacing for a 2D grid with non-Cartesian coordinates
    by considering a single cell at the center of the grid.
    """
    latitudes = np.radians(latitudes)
    longitudes = np.radians(longitudes)

    # Find the center cell index
    center_row, center_col = latitudes.shape[0] // 2, latitudes.shape[1] // 2

    if np.isnan(latitudes[center_row, center_col]):
        return 500

    # Get the latitude and longitude of the center cell
    center_lat, center_lon = latitudes[center_row, center_col], longitudes[center_row, center_col]

    # Calculate the distance from the center cell to its surrounding neighboring cells
    distances = haversine(center_lat, center_lon, latitudes, longitudes)

    # Exclude the center cell distance to itself
    distances[center_row, center_col] = np.inf

    # Calculate the average grid spacing
    avg_spacing = np.nanmean(distances[~np.isinf(distances)])

    return avg_spacing * 1.5 * 1000


def find_index(grid):
    rows, cols = grid.shape
    for i in range(rows - 1):
        for j in range(cols - 1):
            if (not np.isnan(grid[i, j])
                    and not np.isnan(grid[i, j + 1])
                    and not grid[i, j] == 0
                    and not grid[i + 1, j + 1] == 0):
                return i, j
    raise ValueError("No valid grid cell available.")


def center_grid_spacing(x, y):
    i, j = find_index(x)
    spacing = np.mean([abs(x[i + 1, j + 1] - x[i, j]),
                      abs(y[i + 1, j + 1] - y[i, j])]) * 1.5
    return spacing


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


def months_between_dates(start, end, max_months=1200):
    months = []
    current = start.replace(day=1)  # Start from the first day of the month
    while current <= end and max_months > 0:
        months.append(current)
        current = current + relativedelta(months=1)
        max_months = max_months - 1
    return months


def identify_projection(x, y):
    if -180 <= x <= 180 and -180 <= y <= 180:
        return "WGS84"
    elif 420000 <= x <= 900000 and 30000 <= y <= 350000:
        return "CH1903"
    elif 2420000 <= x <= 2900000 and 1030000 <= y <= 1350000:
        return "CH1903+"
    else:
        return "UTM"


def latlng_to_projection(lat, lng, projection):
    lat = np.array(lat)
    lng = np.array(lng)
    if projection == "UTM":
        x, y, zone_number, zone_letter = latlng_to_utm(lat, lng)
    elif projection == "CH1903":
        x, y = latlng_to_ch1903(lat, lng)
    elif projection == "CH1903+":
        x, y = latlng_to_ch1903_plus(lat, lng)
    else:
        raise ValueError('Projection {} unrecognised.'.format(projection))
    return x, y


def projection_to_latlng(x, y, projection):
    x = np.array(x)
    y = np.array(y)
    if projection == "UTM":
        lat, lng = utm_to_latlng(x, y, 32, zone_letter="T")
    elif projection == "CH1903":
        lat, lng = ch1903_to_latlng(x, y)
    elif projection == "CH1903+":
        lat, lng = ch1903_plus_to_latlng(x, y)
    else:
        raise ValueError('Projection {} unrecognised.'.format(projection))
    return lat, lng


def coordinates_to_latlng(x, y):
    x = np.asarray(x).astype(np.float64)
    y = np.asarray(y).astype(np.float64)
    x[x == 0.] = np.nan
    y[y == 0.] = np.nan
    # Detect coordinate system from values
    x_example = x[~np.isnan(x)][0]
    y_example = y[~np.isnan(y)][0]
    if -180 <= x_example <= 180 and -180 <= y_example <= 180:
        # WGS84
        return x, y
    elif 420000 <= x_example <= 900000 and 30000 <= y_example <= 350000:
        # CH1903
        lat, lng = ch1903_to_latlng(x, y)
        return lat, lng
    else:
        # UTM - Default
        x_nan = x[~np.isnan(x)]
        y_nan = y[~np.isnan(x)]
        lat_out = np.zeros(x.shape)
        lng_out = np.zeros(x.shape)
        lat_out[:] = np.nan
        lng_out[:] = np.nan
        lat, lng = utm_to_latlng(x_nan, y_nan, 32, "T")
        lat_out[~np.isnan(x)] = lat
        lng_out[~np.isnan(x)] = lng
        return lat_out, lng_out


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


def latlng_to_ch1903_plus(lat, lng):
    lat = lat * 3600
    lng = lng * 3600
    lat_aux = (lat - 169028.66) / 10000
    lng_aux = (lng - 26782.5) / 10000
    x = 2600072.37 + 211455.93 * lng_aux - 10938.51 * lng_aux * lat_aux - 0.36 * lng_aux * lat_aux ** 2 - 44.54 * lng_aux ** 3
    y = 1200147.07 + 308807.95 * lat_aux + 3745.25 * lng_aux ** 2 + 76.63 * lat_aux ** 2 - 194.56 * lng_aux ** 2 * lat_aux + 119.79 * lat_aux ** 3
    return x, y


def ch1903_plus_to_latlng(x, y):
    x_aux = (x - 2600000) / 1000000
    y_aux = (y - 1200000) / 1000000
    lat = 16.9023892 + 3.238272 * y_aux - 0.270978 * x_aux ** 2 - 0.002528 * y_aux ** 2 - 0.0447 * x_aux ** 2 * y_aux - 0.014 * y_aux ** 3
    lng = 2.6779094 + 4.728982 * x_aux + 0.791484 * x_aux * y_aux + 0.1306 * x_aux * y_aux ** 2 - 0.0436 * x_aux ** 3
    lat = (lat * 100) / 36
    lng = (lng * 100) / 36
    return lat, lng


def utm_to_latlng(easting, northing, zone_number, zone_letter=None, northern=None, strict=True):
    K0 = 0.9996

    E = 0.00669438
    E2 = E * E
    E3 = E2 * E
    E_P2 = E / (1 - E)

    SQRT_E = np.sqrt(1 - E)
    _E = (1 - SQRT_E) / (1 + SQRT_E)
    _E2 = _E * _E
    _E3 = _E2 * _E
    _E4 = _E3 * _E
    _E5 = _E4 * _E

    M1 = (1 - E / 4 - 3 * E2 / 64 - 5 * E3 / 256)

    P2 = (3 / 2 * _E - 27 / 32 * _E3 + 269 / 512 * _E5)
    P3 = (21 / 16 * _E2 - 55 / 32 * _E4)
    P4 = (151 / 96 * _E3 - 417 / 128 * _E5)
    P5 = (1097 / 512 * _E4)
    R = 6378137

    if not zone_letter and northern is None:
        raise ValueError('either zone_letter or northern needs to be set')

    elif zone_letter and northern is not None:
        raise ValueError('set either zone_letter or northern, but not both')

    if strict:
        if not in_bounds(easting, 100000, 1000000, upper_strict=True):
            raise ValueError('easting {} out of range (must be between 100,000 m and 999,999 m)'.format(easting))
        if not in_bounds(northing, 0, 10000000):
            raise ValueError('northing {} out of range (must be between 0 m and 10,000,000 m)'.format(northing))

    check_valid_zone(zone_number, zone_letter)

    if zone_letter:
        zone_letter = zone_letter.upper()
        northern = (zone_letter >= 'N')

    x = easting - 500000
    y = northing

    if not northern:
        y -= 10000000

    m = y / K0
    mu = m / (R * M1)

    p_rad = (mu +
             P2 * np.sin(2 * mu) +
             P3 * np.sin(4 * mu) +
             P4 * np.sin(6 * mu) +
             P5 * np.sin(8 * mu))

    p_sin = np.sin(p_rad)
    p_sin2 = p_sin * p_sin

    p_cos = np.cos(p_rad)

    p_tan = p_sin / p_cos
    p_tan2 = p_tan * p_tan
    p_tan4 = p_tan2 * p_tan2

    ep_sin = 1 - E * p_sin2
    ep_sin_sqrt = np.sqrt(1 - E * p_sin2)

    n = R / ep_sin_sqrt
    r = (1 - E) / ep_sin

    c = E_P2 * p_cos ** 2
    c2 = c * c

    d = x / (n * K0)
    d2 = d * d
    d3 = d2 * d
    d4 = d3 * d
    d5 = d4 * d
    d6 = d5 * d

    latitude = (p_rad - (p_tan / r) *
                (d2 / 2 -
                 d4 / 24 * (5 + 3 * p_tan2 + 10 * c - 4 * c2 - 9 * E_P2)) +
                d6 / 720 * (61 + 90 * p_tan2 + 298 * c + 45 * p_tan4 - 252 * E_P2 - 3 * c2))

    longitude = (d -
                 d3 / 6 * (1 + 2 * p_tan2 + c) +
                 d5 / 120 * (5 - 2 * c + 28 * p_tan2 - 3 * c2 + 8 * E_P2 + 24 * p_tan4)) / p_cos

    longitude = mod_angle(longitude + np.radians(zone_number_to_central_longitude(zone_number)))

    return (np.degrees(latitude),
            np.degrees(longitude))


def latlng_to_utm(latitude, longitude, force_zone_number=None, force_zone_letter=None):
    K0 = 0.9996

    E = 0.00669438
    E2 = E * E
    E3 = E2 * E
    E_P2 = E / (1 - E)

    SQRT_E = np.sqrt(1 - E)
    _E = (1 - SQRT_E) / (1 + SQRT_E)
    _E2 = _E * _E
    _E3 = _E2 * _E
    _E4 = _E3 * _E
    _E5 = _E4 * _E

    M1 = (1 - E / 4 - 3 * E2 / 64 - 5 * E3 / 256)
    M2 = (3 * E / 8 + 3 * E2 / 32 + 45 * E3 / 1024)
    M3 = (15 * E2 / 256 + 45 * E3 / 1024)
    M4 = (35 * E3 / 3072)
    R = 6378137

    if not in_bounds(latitude, -80, 84):
        raise ValueError('latitude out of range (must be between 80 deg S and 84 deg N)')
    if not in_bounds(longitude, -180, 180):
        raise ValueError('longitude out of range (must be between 180 deg W and 180 deg E)')
    if force_zone_number is not None:
        check_valid_zone(force_zone_number, force_zone_letter)

    lat_rad = np.radians(latitude)
    lat_sin = np.sin(lat_rad)
    lat_cos = np.cos(lat_rad)

    lat_tan = lat_sin / lat_cos
    lat_tan2 = lat_tan * lat_tan
    lat_tan4 = lat_tan2 * lat_tan2

    if force_zone_number is None:
        zone_number = latlon_to_zone_number(latitude, longitude)
    else:
        zone_number = force_zone_number

    if force_zone_letter is None:
        zone_letter = latitude_to_zone_letter(latitude)
    else:
        zone_letter = force_zone_letter

    lon_rad = np.radians(longitude)
    central_lon = zone_number_to_central_longitude(zone_number)
    central_lon_rad = np.radians(central_lon)

    n = R / np.sqrt(1 - E * lat_sin ** 2)
    c = E_P2 * lat_cos ** 2

    a = lat_cos * mod_angle(lon_rad - central_lon_rad)
    a2 = a * a
    a3 = a2 * a
    a4 = a3 * a
    a5 = a4 * a
    a6 = a5 * a

    m = R * (M1 * lat_rad -
             M2 * np.sin(2 * lat_rad) +
             M3 * np.sin(4 * lat_rad) -
             M4 * np.sin(6 * lat_rad))

    easting = K0 * n * (a +
                        a3 / 6 * (1 - lat_tan2 + c) +
                        a5 / 120 * (5 - 18 * lat_tan2 + lat_tan4 + 72 * c - 58 * E_P2)) + 500000

    northing = K0 * (m + n * lat_tan * (a2 / 2 +
                                        a4 / 24 * (5 - lat_tan2 + 9 * c + 4 * c ** 2) +
                                        a6 / 720 * (61 - 58 * lat_tan2 + lat_tan4 + 600 * c - 330 * E_P2)))

    if mixed_signs(latitude):
        raise ValueError("latitudes must all have the same sign")
    elif negative(latitude):
        northing += 10000000

    return easting, northing, zone_number, zone_letter


def in_bounds(x, lower, upper, upper_strict=False):
    if upper_strict:
        return lower <= np.min(x) and np.max(x) < upper
    else:
        return lower <= np.min(x) and np.max(x) <= upper


def check_valid_zone(zone_number, zone_letter):
    if not 1 <= zone_number <= 60:
        raise ValueError('zone number out of range (must be between 1 and 60)')

    if zone_letter:
        zone_letter = zone_letter.upper()

        if not 'C' <= zone_letter <= 'X' or zone_letter in ['I', 'O']:
            raise ValueError('zone letter out of range (must be between C and X)')


def mixed_signs(x):
    return np.min(x) < 0 and np.max(x) >= 0


def negative(x):
    return np.max(x) < 0


def mod_angle(value):
    """Returns angle in radians to be between -pi and pi"""
    return (value + np.pi) % (2 * np.pi) - np.pi


def latitude_to_zone_letter(latitude):
    zone_letters = "CDEFGHJKLMNPQRSTUVWXX"
    if isinstance(latitude, np.ndarray):
        latitude = latitude.flat[0]

    if -80 <= latitude <= 84:
        return zone_letters[int(latitude + 80) >> 3]
    else:
        return None


def latlon_to_zone_number(latitude, longitude):
    if isinstance(latitude, np.ndarray):
        latitude = latitude.flat[0]
    if isinstance(longitude, np.ndarray):
        longitude = longitude.flat[0]

    if 56 <= latitude < 64 and 3 <= longitude < 12:
        return 32

    if 72 <= latitude <= 84 and longitude >= 0:
        if longitude < 9:
            return 31
        elif longitude < 21:
            return 33
        elif longitude < 33:
            return 35
        elif longitude < 42:
            return 37

    return int((longitude + 180) / 6) + 1


def zone_number_to_central_longitude(zone_number):
    return (zone_number - 1) * 6 - 180 + 3
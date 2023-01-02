import os
import shutil
import requests
import numpy as np
from fastapi import HTTPException
from datetime import datetime, timedelta


def convert_to_unit(time, units):
    if units == "seconds since 2008-03-01 00:00:00":
        return (time - datetime(2008, 3, 1)).total_seconds()
    elif units == "seconds since 1970-01-01 00:00:00":
        return time.timestamp()
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies unable to read NetCDF with time unit: {}".format(units))


def convert_from_unit(time, units):
    if units == "seconds since 2008-03-01 00:00:00":
        return datetime.utcfromtimestamp(time + (datetime(2008, 3, 1) - datetime(1970, 1, 1)).total_seconds())
    elif units == "seconds since 1970-01-01 00:00:00":
        return datetime.utcfromtimestamp(time)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies unable to read NetCDF with time unit: {}".format(units))


def get_closest_index(value, array):
    array = np.sort(np.asarray(array))
    if len(array) == 0:
        raise ValueError("Array must be longer than len(0) to find index of value")
    elif len(array) == 1:
        return 0
    if value > (2 * array[-1] - array[-2]):
        raise HTTPException(status_code=400,
                            detail="Value {} greater than max available ({})".format(value, array[-1]))
    elif value < (2 * array[0] - array[-1]):
        raise HTTPException(status_code=400,
                            detail="Value {} less than min available ({})".format(value, array[0]))
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
    

import os
import shutil
import requests
import numpy as np
from datetime import timedelta


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


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
    

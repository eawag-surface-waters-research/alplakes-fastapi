import os
import netCDF4
from enum import Enum
from pydantic import BaseModel
from fastapi import HTTPException
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta, SU

from app import functions


class Models(str, Enum):
    delft3dflow = "delft3d-flow"


class Lakes(str, Enum):
    geneva = "geneva"
    greifensee = "greifensee"
    zurich = "zurich"
    biel = "biel"


def verify_simulations_layer(model, lake, datetime, depth):
    return True


def get_simulations_layer(filesystem, model, lake, time, depth):
    if model == "delft3d-flow":
        get_simulations_layer_delft3dflow(filesystem, lake, time, depth)
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies data is not available for {}".format(model))


def get_simulations_layer_delft3dflow(filesystem, lake, time, depth):
    model = "delft3d-flow"
    origin = datetime.strptime(time, "%Y%m%d%H")
    last_sunday = origin + relativedelta(weekday=SU(-1))
    previous_sunday = last_sunday - timedelta(days=7)
    lakes = os.path.join(filesystem, "media/simulations", model, "results")
    if not os.path.isdir(os.path.join(lakes, lake)):
        raise HTTPException(status_code=400,
                            detail="{} simulation results are not available for {} please select from: [{}]"
                            .format(model, lake, ", ".join(os.listdir(lakes))))
    if os.path.isfile(os.path.join(lakes, lake, "{}.nc".format(last_sunday.strftime("%Y%m%d")))):
        file = os.path.join(lakes, lake, "{}.nc".format(last_sunday.strftime("%Y%m%d")))
    elif os.path.isfile(os.path.join(lakes, lake, "{}.nc".format(previous_sunday.strftime("%Y%m%d")))):
        file = os.path.join(lakes, lake, "{}.nc".format(previous_sunday.strftime("%Y%m%d")))
    else:
        raise HTTPException(status_code=400,
                            detail="Apologies data is not available for {} at {}".format(lake, time))
    with netCDF4.Dataset(file) as nc:
        print(nc)


class Notification(BaseModel):
    type: str
    value: str


def notify(filesystem, notification):
    n = notification.dict()
    if n["type"] == "new":
        return notify_new(filesystem, n["value"])
    else:
        raise HTTPException(status_code=400,
                            detail="The following notification type is not recognised: {}".format(n["type"]))


def notify_new(filesystem, file):
    accepted_buckets = ["https://alplakes-eawag.s3"]
    if file[:25] in accepted_buckets:
        u = file.split("_")
        s = file.split("/")
        local = os.path.join(filesystem, "media/simulations", s[-3], "results", u[-2], u[-1])
        functions.download_file(file, local)
        return "Successfully downloaded {} to API storage.".format(file)
    else:
        raise HTTPException(status_code=500,
                            detail="Only new simulation in the Alplakes S3 Bucket are accepted.")

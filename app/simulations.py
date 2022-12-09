import os
from pydantic import BaseModel
from fastapi import HTTPException

from app import functions


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
    prefix = "https://alplakes-eawag.s3.eu-central-1.amazonaws.com/simulations/"
    if file[:31] == "s3://alplakes-eawag/simulations":
        path = file[32:]
        url = prefix + path
        p1 = path.split("/")
        model = p1[0]
        p2 = p1[-1].split("_")
        lake = p2[-3]
        filename = "{}.nc".format(p2[-2])
        local = os.path.join(filesystem, "media/simulations", model, "results", lake, filename)
        functions.download_file(url, local)
        return "Successfully downloaded {} to API storage.".format(file)
    else:
        raise HTTPException(status_code=500,
                            detail="Only new simulation in the Alplakes S3 Bucket are accepted.")

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

import os
import json
import shutil
import netCDF4
import logging
import numpy as np
from enum import Enum
from pydantic import BaseModel
from fastapi import HTTPException
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta, SU

from app import functions


class Satellites(str, Enum):
    sentinel2 = "sentinel2"
    sentinel3 = "sentinel3"


class Parameters(str, Enum):
    chla = "chla"
    tsm = "tsm"
    whiting = "whiting"
    primaryproduction = "primaryproduction"
    secchi = "secchi"
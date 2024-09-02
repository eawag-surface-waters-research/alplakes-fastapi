import json
import time
import numpy as np
import seaborn as sns
from datetime import  datetime
import matplotlib.pyplot as plt

import meteoswiss
import bafu
import insitu
import simulations

filesystem = "../filesystem"

function = "meteoswiss.get_meteodata_measured"

if function == "meteoswiss.get_cosmo_metadata":
    data = meteoswiss.get_cosmo_metadata(filesystem)
    print(data)

if function == "meteoswiss.get_cosmo_area_reanalysis":
    data = meteoswiss.get_cosmo_area_reanalysis(filesystem, "VNXQ34", ["T_2M"], "20221231", "20230101", 47.236215858048524, 8.526197876275669, 47.456287451632484, 8.835863473096651)
    print(data)

if function == "meteoswiss.get_cosmo_area_forecast":
    data = meteoswiss.get_cosmo_area_forecast(filesystem, "VNXZ32", ["T_2M"], "20230101", 46.156, 6.134, 46.54, 6.957)
    print(data)

if function == "meteoswiss.get_cosmo_point_reanalysis":
    data = meteoswiss.get_cosmo_point_reanalysis(filesystem, "VNXQ34", ["T_2M"], "20221231", "20230101", 46.5, 6.67)
    print(data)

if function == "meteoswiss.get_cosmo_point_forecast":
    data = meteoswiss.get_cosmo_point_forecast(filesystem, "VNXZ32", ["T_2M"], "20230101", 46.5, 6.67)
    print(data)

if function == "meteoswiss.get_icon_metadata":
    data = meteoswiss.get_icon_metadata(filesystem)
    print(data)

if function == "meteoswiss.get_icon_area_reanalysis":
    data = meteoswiss.get_icon_area_reanalysis(filesystem, "kenda-ch1", ["T_2M"], "20240729", "20240729", 47.236215858048524, 8.526197876275669, 47.456287451632484, 8.835863473096651)
    print(data)

if function == "meteoswiss.get_icon_area_forecast":
    data = meteoswiss.get_icon_area_forecast(filesystem, "icon-ch2-eps", ["T_2M"], "20240703", 46.156, 6.134, 46.54, 6.957)
    print(data["time"])

if function == "meteoswiss.get_icon_point_reanalysis":
    data = meteoswiss.get_icon_point_reanalysis(filesystem, "kenda-ch1", ["T_2M"], "20240729", "20240729", 46.5, 6.67)
    print(data)

if function == "meteoswiss.get_icon_point_forecast":
    data = meteoswiss.get_icon_point_forecast(filesystem, "icon-ch2-eps", ["T_2M"], "20240703", 46.5, 6.67)
    print(data)

if function == "meteoswiss.get_meteodata_station_metadata":
    data = meteoswiss.get_meteodata_station_metadata(filesystem, "PUY")
    print(data)

if function == "meteoswiss.get_meteodata_measured":
    data = meteoswiss.get_meteodata_measured(filesystem, "PUY", "pva200h0", "20230101", "20240210")
    plt.plot(data["time"], data["pva200h0"])
    plt.show()



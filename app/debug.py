import json
import time
import numpy as np
import seaborn as sns
import pandas as pd
from datetime import  datetime
import matplotlib.pyplot as plt

import meteoswiss
import bafu
import insitu
import simulations

filesystem = "../filesystem"

function = "meteoswiss.get_cosmo_point_forecast"

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
    plt.plot(data["time"], data["T_2M"]["data"])
    plt.show()

if function == "meteoswiss.get_cosmo_point_forecast":
    data = meteoswiss.get_cosmo_point_forecast(filesystem, "VNXZ32", ["T_2M"], "20230101", 46.5, 6.67)
    plt.plot(data["time"], data["T_2M"]["data"])
    plt.show()

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

if function == "bafu.get_hydrodata_station_metadata":
    data = bafu.get_hydrodata_station_metadata(filesystem, 2009)
    print(data)

if function == "bafu.get_hydrodata_measured":
    data = bafu.get_hydrodata_measured(filesystem, 2009, "AbflussPneumatikunten", "20210207", "20230201")
    plt.plot(data["time"], data["AbflussPneumatikunten"])
    plt.show()

if function == "insitu.get_insitu_secchi_metadata":
    data = insitu.get_insitu_secchi_metadata(filesystem)
    print(data)

if function == "insitu.get_insitu_secchi_lake":
    data = insitu.get_insitu_secchi_lake(filesystem, "geneva")
    plt.plot(data["time"], data["secchi"]["data"])
    plt.show()

if function == "simulations.get_metadata":
    data = simulations.get_metadata(filesystem)
    print(data)

if function == "simulations.get_metadata_lake":
    data = simulations.get_metadata_lake(filesystem, "delft3d-flow", "geneva")
    print(data)

if function == "simulations.get_simulations_point":
    data = simulations.get_simulations_point(filesystem, "delft3d-flow", "geneva", "202304050300", "202304112300", 1, 46.5, 6.67)
    plt.plot([datetime.fromisoformat(t) for t in data["time"]], data["temperature"]["data"])
    plt.show()

if function == "simulations.get_simulations_layer":
    data = simulations.get_simulations_layer(filesystem, "delft3d-flow", "geneva", "202304050300", 1)
    temperature = pd.DataFrame(data["temperature"]["data"]).apply(pd.to_numeric, errors='coerce').to_numpy()
    plt.imshow(temperature, cmap='viridis', interpolation='none')
    plt.show()

if function == "simulations.get_simulations_layer_average_temperature":
    data = simulations.get_simulations_layer_average_temperature(filesystem, "delft3d-flow", "geneva", "202304050300", "202304112300", 1)
    plt.plot([datetime.fromisoformat(t) for t in data["time"]], data["temperature"]["data"])
    plt.show()

if function == "simulations.get_simulations_profile":
    data = simulations.get_simulations_profile(filesystem, "delft3d-flow", "geneva", "202304050300", 46.5, 6.67)
    plt.plot(data["temperature"]["data"], np.array(data["depth"]["data"])*-1)
    plt.show()

if function == "simulations.get_simulations_transect":
    data = simulations.get_simulations_transect(filesystem, "delft3d-flow", "geneva", "202304050300", "46.37,46.54", "6.56,6.54")
    temperature = pd.DataFrame(data["temperature"]["data"]).apply(pd.to_numeric, errors='coerce').to_numpy()
    plt.imshow(temperature, cmap='viridis', interpolation='none')
    plt.show()

if function == "simulations.get_simulations_transect_period":
    data = simulations.get_simulations_transect_period(filesystem, "delft3d-flow", "geneva", "202304050300", "202304051200", "46.37,46.54", "6.56,6.54")
    temperature = pd.DataFrame(data["temperature"]["data"][0]).apply(pd.to_numeric, errors='coerce').to_numpy()
    plt.imshow(temperature, cmap='viridis', interpolation='none')
    plt.show()

if function == "simulations.get_simulations_depthtime":
    data = simulations.get_simulations_depthtime(filesystem, "delft3d-flow", "geneva", "202304050300", "202304112300", 46.5, 6.67)
    temperature = pd.DataFrame(data["temperature"]["data"]).apply(pd.to_numeric, errors='coerce').to_numpy()
    plt.imshow(temperature, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.show()

if function == "simulations.get_one_dimensional_metadata":
    data = simulations.get_one_dimensional_metadata(filesystem)
    print(data)

if function == "simulations.get_one_dimensional_metadata_lake":
    data = simulations.get_one_dimensional_metadata_lake(filesystem, "simstrat", "geneva")
    print(data)

if function == "simulations.get_one_dimensional_point":
    data = simulations.get_one_dimensional_point(filesystem, "simstrat", "aegeri", "T", "202405050300","202406072300",1)
    plt.plot([datetime.fromisoformat(t) for t in data["time"]], data["T"]["data"])
    plt.show()

if function == "simulations.get_one_dimensional_depth_time":
    data = simulations.get_one_dimensional_depth_time(filesystem, "simstrat", "aegeri", "T", "202405050300","202406072300")
    temperature = pd.DataFrame(data["T"]["data"]).apply(pd.to_numeric, errors='coerce').to_numpy()
    plt.imshow(temperature, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.show()

if function == "simulations.get_one_dimensional_day_of_year":
    simulations.write_one_dimensional_day_of_year_simstrat(filesystem, "aegeri", "T", 1)
    out = simulations.get_one_dimensional_day_of_year_simstrat(filesystem, "aegeri", "T", 1)
    plt.plot(out["mean"], label="mean")
    plt.plot(out["max"], label="max")
    plt.plot(out["min"], label="min")
    plt.plot(out["lastyear"], label="last")
    plt.fill_between(range(366), out["perc5"], out["perc95"], color='lightcoral', alpha=0.5)
    plt.fill_between(range(366), out["perc25"], out["perc75"], color='skyblue', alpha=0.5)
    plt.legend()
    plt.show()

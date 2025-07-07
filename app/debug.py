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

function = "simulations.get_simulations_layer_mitgcm"

if function == "meteoswiss.get_cosmo_metadata":
    data = meteoswiss.get_cosmo_metadata(filesystem)
    print(data)

if function == "meteoswiss.get_cosmo_area_reanalysis":
    data = meteoswiss.get_cosmo_area_reanalysis(filesystem, "VNXQ34", ["T_2M"], "20221231", "20230101", 47.236215858048524, 8.526197876275669, 47.456287451632484, 8.835863473096651)
    print(data)

if function == "meteoswiss.get_cosmo_area_forecast":
    data = meteoswiss.get_cosmo_area_forecast(filesystem, "VNXZ32", ["T_2M"], "20230101", 46.156, 6.134, 46.54, 6.957)
    plt.plot(data["time"], data["variables"]["T_2M"]["data"])
    plt.show()
    print(data)

if function == "meteoswiss.get_cosmo_point_reanalysis":
    data = meteoswiss.get_cosmo_point_reanalysis(filesystem, "VNXQ34", ["T_2M"], "20221231", "20230101", 46.5, 6.67)
    plt.plot(data["time"], data["variables"]["T_2M"]["data"])
    plt.show()

if function == "meteoswiss.get_cosmo_point_forecast":
    data = meteoswiss.get_cosmo_point_forecast(filesystem, "VNXZ32", ["T_2M"], "20230101", 46.5, 6.67)
    plt.plot(data["time"], data["variables"]["T_2M"]["data"])
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
    data = meteoswiss.get_meteodata_measured(filesystem, "PUY", ["pva200h0"], "20230101", "20240210")
    plt.plot(data["time"], data["variables"]["pva200h0"]["data"])
    plt.show()

if function == "bafu.get_hydrodata_station_metadata":
    data = bafu.get_hydrodata_station_metadata(filesystem, 2009)
    print(data)

if function == "bafu.get_hydrodata_measured":
    data = bafu.get_hydrodata_measured(filesystem, 2009, "AbflussPneumatikunten", "20210207", "20230201")
    plt.plot(data["time"], data["variable"]["data"])
    plt.show()

if function == "insitu.get_insitu_secchi_metadata":
    data = insitu.get_insitu_secchi_metadata(filesystem)
    print(data)

if function == "insitu.get_insitu_secchi_lake":
    data = insitu.get_insitu_secchi_lake(filesystem, "geneva")
    plt.plot(data["time"], data["variable"]["data"])
    plt.show()

if function == "simulations.get_metadata":
    data = simulations.get_metadata(filesystem)
    print(data)

if function == "simulations.get_metadata_lake":
    data = simulations.get_metadata_lake(filesystem, "mitgcm", "zurich")
    print(data)

if function == "simulations.get_simulations_point":
    data = simulations.get_simulations_point(filesystem, "delft3d-flow", "geneva", "202304050300", "202304172300", 1, 46.5, 6.67, ["temperature", "velocity"])
    plt.plot(data["time"], data["variables"]["temperature"]["data"])
    plt.show()

if function == "simulations.get_simulations_point_mitgcm":
    data = simulations.get_simulations_point(filesystem, "mitgcm", "zurich", "202506200300", "202507042300", 1, 47.22, 8.72, ["temperature", "velocity"])
    plt.plot(data["time"], data["variables"]["temperature"]["data"])
    plt.show()

if function == "simulations.get_simulations_layer_delft3d":
    data = simulations.get_simulations_layer(filesystem, "delft3d-flow", "geneva", "202304050300", 1, ["temperature"])
    temperature = pd.DataFrame(data["variables"]["temperature"]["data"]).apply(pd.to_numeric, errors='coerce').to_numpy()
    plt.imshow(temperature, cmap='viridis', interpolation='none')
    plt.show()

if function == "simulations.get_simulations_layer_mitgcm":
    data = simulations.get_simulations_layer(filesystem, "mitgcm", "zurich", "202507020300", 1, ["temperature"])
    temperature = pd.DataFrame(data["variables"]["temperature"]["data"]).apply(pd.to_numeric, errors='coerce').to_numpy()
    plt.imshow(temperature, cmap='viridis', interpolation='none')
    plt.show()

if function == "simulations.get_simulations_layer_alplakes_mitgcm":
    data = simulations.get_simulations_layer_alplakes(filesystem, "mitgcm", "zurich", "temperature", "202507020300", "202507040300", 1)
    print(data)

if function == "simulations.get_simulations_layer_average_temperature":
    data = simulations.get_simulations_layer_average_temperature(filesystem, "delft3d-flow", "geneva", "202304050300", "202304112300", 1)
    plt.plot(data["time"], data["variables"]["temperature"]["data"])
    plt.show()

if function == "simulations.get_simulations_average_bottom_temperature":
    data = simulations.get_simulations_average_bottom_temperature(filesystem, "delft3d-flow", "geneva",
                                                                 "202304050300", "202304112300")
    temperature = np.array(data["variable"]["data"], dtype=float)
    plt.imshow(temperature, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.show()

if function == "simulations.get_simulations_profile":
    data = simulations.get_simulations_profile(filesystem, "delft3d-flow", "geneva", "202304050300", 46.5, 6.67, ["temperature"])
    plt.plot(data["variables"]["temperature"]["data"], np.array(data["depth"]["data"])*-1)
    plt.show()

if function == "simulations.get_simulations_depthtime":
    data = simulations.get_simulations_depthtime(filesystem, "delft3d-flow", "geneva", "202304050300", "202304112300", 46.5, 6.67, ["temperature", "velocity"])
    temperature = pd.DataFrame(data["variables"]["temperature"]["data"]).apply(pd.to_numeric, errors='coerce').to_numpy()
    plt.imshow(temperature, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.show()
    plt.plot(np.array(data["variables"]["temperature"]["data"])[:, 0], np.array(data["depth"]["data"]) * -1)
    plt.show()

if function == "simulations.get_simulations_transect":
    data = simulations.get_simulations_transect(filesystem, "delft3d-flow", "geneva", "202304050300", "46.37,46.54", "6.56,6.54", ["temperature", "velocity"])
    temperature = pd.DataFrame(data["variables"]["temperature"]["data"]).apply(pd.to_numeric, errors='coerce').to_numpy()
    plt.imshow(temperature, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.show()

if function == "simulations.get_simulations_transect_period":
    data = simulations.get_simulations_transect_period(filesystem, "delft3d-flow", "garda", "202312050000", "202312150000", "45.435,45.589,45.719", "10.687,10.635,10.673", ["temperature", "velocity"])
    for i in range(len(data["variables"]["temperature"]["data"])):
        temperature = pd.DataFrame(data["variables"]["temperature"]["data"][i]).apply(pd.to_numeric, errors='coerce').to_numpy()
        plt.imshow(temperature, cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title(data["time"][i])
        plt.show()

if function == "simulations.get_one_dimensional_metadata":
    data = simulations.get_one_dimensional_metadata(filesystem)
    print(data)

if function == "simulations.get_one_dimensional_metadata_lake":
    data = simulations.get_one_dimensional_metadata_lake(filesystem, "simstrat", "geneva")
    print(data)

if function == "simulations.get_one_dimensional_point":
    data = simulations.get_one_dimensional_point(filesystem, "simstrat", "aegeri", "202405050300","202406072300",1, ["T"])
    plt.plot(data["time"], data["variables"]["T"]["data"])
    plt.show()

if function == "simulations.get_one_dimensional_profile":
    data = simulations.get_one_dimensional_profile(filesystem, "simstrat", "aegeri", "202405050300",["T"])
    plt.plot(data["variables"]["T"]["data"], np.array(data["depth"]["data"]) * -1)
    plt.show()

if function == "simulations.get_one_dimensional_depth_time":
    data = simulations.get_one_dimensional_depth_time(filesystem, "simstrat", "aegeri", "202405050300","202406072300", ["T"])
    temperature = pd.DataFrame(data["variables"]["T"]["data"]).apply(pd.to_numeric, errors='coerce').to_numpy()
    plt.imshow(temperature, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.show()

if function == "simulations.get_one_dimensional_day_of_year_metadata":
    data = simulations.get_one_dimensional_day_of_year_metadata(filesystem)
    print(data)

if function == "simulations.get_one_dimensional_day_of_year":
    simulations.write_one_dimensional_day_of_year_simstrat(filesystem, "aegeri", "T", 1.0)
    data = simulations.get_one_dimensional_day_of_year_simstrat(filesystem, "aegeri", "T", 1.0)
    plt.plot(data["variables"]["mean"]["data"], label="mean")
    plt.plot(data["variables"]["max"]["data"], label="max")
    plt.plot(data["variables"]["min"]["data"], label="min")
    plt.plot(data["variables"]["lastyear"]["data"], label="last")
    plt.fill_between(range(366), data["variables"]["perc5"]["data"], data["variables"]["perc95"]["data"], color='lightcoral', alpha=0.5)
    plt.fill_between(range(366), data["variables"]["perc25"]["data"], data["variables"]["perc75"]["data"], color='skyblue', alpha=0.5)
    plt.legend()
    plt.show()

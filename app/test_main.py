from fastapi.testclient import TestClient
from datetime import datetime
import pytest
from .main import app

client = TestClient(app)


def test_welcome():
    response = client.get("/")
    assert response.status_code == 200


def test_meteoswiss_cosmo_metadata():
    response = client.get("/meteoswiss/cosmo/metadata")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_cosmo_area_reanalysis():
    response = client.get("/meteoswiss/cosmo/area/reanalysis/VNXQ34/20230101/20230101/46.49/6.65/46.51/6.67")
    assert response.status_code == 200
    data = response.json()
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")
    assert "lat" in data
    assert "lng" in data


def test_get_cosmo_area_forecast():
    response = client.get("/meteoswiss/cosmo/area/forecast/VNXZ32/20230101/46.49/6.65/46.51/6.67")
    assert response.status_code == 200
    data = response.json()
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")
    assert "lat" in data
    assert "lng" in data


def test_get_cosmo_point_reanalysis():
    response = client.get("/meteoswiss/cosmo/point/reanalysis/VNXQ34/20230101/20230101/46.5/6.67")
    assert response.status_code == 200
    data = response.json()
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")
    assert "lat" in data
    assert "lng" in data


def test_get_cosmo_point_forecast():
    response = client.get("/meteoswiss/cosmo/point/forecast/VNXZ32/20230101/46.5/6.67")
    assert response.status_code == 200
    data = response.json()
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")
    assert "lat" in data
    assert "lng" in data


def test_meteoswiss_icon_metadata():
    response = client.get("/meteoswiss/icon/metadata")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_icon_area_reanalysis():
    response = client.get("/meteoswiss/icon/area/reanalysis/kenda-ch1/20240729/20240729/46.49/6.65/46.51/6.67")
    assert response.status_code == 200
    data = response.json()
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")
    assert "lat" in data
    assert "lng" in data


def test_icon_area_forecast():
    response = client.get("/meteoswiss/icon/area/forecast/icon-ch2-eps/20240703/46.49/6.65/46.51/6.67")
    assert response.status_code == 200
    data = response.json()
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")
    assert "lat" in data
    assert "lng" in data


def test_icon_point_reanalysis():
    response = client.get("/meteoswiss/icon/point/reanalysis/kenda-ch1/20240729/20240729/46.49/6.65")
    assert response.status_code == 200
    data = response.json()
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")
    assert "lat" in data
    assert "lng" in data


def test_icon_point_forecast():
    response = client.get("/meteoswiss/icon/point/forecast/icon-ch2-eps/20240703/46.49/6.65")
    assert response.status_code == 200
    data = response.json()
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")
    assert "lat" in data
    assert "lng" in data


def test_meteoswiss_meteodata_station_metadata_station():
    response = client.get("/meteoswiss/meteodata/metadata/PUY")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_meteoswiss_meteodata_measured():
    response = client.get("/meteoswiss/meteodata/measured/PUY/20230101/20240210")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["variables"]["air_temperature"]["data"][0], float)
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")


def test_arso_meteodata_station_metadata_station():
    response = client.get("/arso/meteodata/metadata/2213")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_arso_meteodata_measured():
    response = client.get("/arso/meteodata/measured/2213/20230101/20240210")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["variables"]["air_temperature"]["data"][0], float)
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")


def test_thredds_meteodata_station_metadata_station():
    response = client.get("/thredds/meteodata/metadata/73329001")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_thredds_meteodata_measured():
    response = client.get("/thredds/meteodata/measured/73329001/20230101/20240210")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["variables"]["air_temperature"]["data"][0], float)
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")


def test_mistral_meteodata_station_metadata_station():
    response = client.get("/mistral/meteodata/metadata/trn196")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_mistral_meteodata_measured():
    response = client.get("/mistral/meteodata/measured/trn196/20230101/20240210")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["variables"]["air_temperature"]["data"][0], float)
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")


def test_geosphere_meteodata_station_metadata_station():
    response = client.get("/geosphere/meteodata/metadata/6512")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_geosphere_meteodata_measured():
    response = client.get("/geosphere/meteodata/measured/6512/20230101/20240210")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["variables"]["air_temperature"]["data"][0], float)
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")


def test_bafu_hydrodata_station_metadata():
    response = client.get("/bafu/hydrodata/metadata/2009")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_bafu_hydrodata_measured():
    variable = "AbflussPneumatikunten"
    response = client.get("/bafu/hydrodata/measured/2009/{}/20210207/20230201".format(variable))
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["variable"]["data"][0], float)
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")


def test_bafu_hydrodata_measured_resample():
    variable = "AbflussPneumatikunten"
    response = client.get("/bafu/hydrodata/measured/2009/{}/20210207/20230201?resample=hourly".format(variable))
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["variable"]["data"][0], float)
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")
    assert (datetime.strptime(data["time"][1], "%Y-%m-%dT%H:%M:%S%z") -
            datetime.strptime(data["time"][0], "%Y-%m-%dT""%H:%M:%S%z")).total_seconds() == 3600


def test_insitu_secchi_metadata():
    response = client.get("/insitu/secchi/metadata")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_insitu_secchi_lake():
    lake = "geneva"
    response = client.get("/insitu/secchi/{}".format(lake))
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["variable"]["data"][0], float)
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")


def test_simulations_metadata():
    response = client.get("/simulations/metadata")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_simulations_metadata_lake():
    response = client.get("/simulations/metadata/delft3d-flow/geneva")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


def test_simulations_point():
    response = client.get("/simulations/point/delft3d-flow/geneva/202304050300/202304112300/1/46.5/6.67")
    assert response.status_code == 200
    data = response.json()
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")
    assert "lat" in data
    assert "lng" in data


def test_simulations_layer():
    response = client.get("/simulations/layer/delft3d-flow/geneva/202304050300/1")
    assert response.status_code == 200
    data = response.json()
    assert datetime.strptime(data["time"], "%Y-%m-%dT%H:%M:%S%z")
    assert "lat" in data
    assert "lng" in data


def test_simulations_layer_alplakes():
    response = client.get("/simulations/layer_alplakes/delft3d-flow/geneva/temperature/202304050300/202304112300/1")
    assert response.status_code == 200


def test_simulations_layer_average_temperature():
    response = client.get("/simulations/layer/average_temperature/delft3d-flow/geneva/202304050300/202304112300/1")
    assert response.status_code == 200
    data = response.json()
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")


def test_simulations_average_bottom_temperature():
    response = client.get("/simulations/layer/average_bottom_temperature/delft3d-flow/geneva/202304050300/202304112300")
    assert response.status_code == 200
    data = response.json()
    assert "lat" in data
    assert "lng" in data
    assert "variable" in data


def test_simulations_simulations_profile():
    response = client.get("/simulations/profile/delft3d-flow/geneva/202304050300/46.5/6.67")
    assert response.status_code == 200
    data = response.json()
    assert datetime.strptime(data["time"], "%Y-%m-%dT%H:%M:%S%z")


@pytest.mark.parametrize("url", [
    "/simulations/transect/delft3d-flow/geneva/202304030400/46.351,46.294/6.177,6.277",
    "/simulations/transect/delft3d-flow/geneva/202304080400/46.351,46.294,46.351/6.177,6.277,6.177",
    "/simulations/transect/delft3d-flow/garda/202312050000/45.435,45.589,45.719/10.687,10.635,10.673",
], ids=["delft3d-flow_single_file_single_segment_ch1903",
        "delft3d-flow_multiple_files_multiple_segments_ch1903",
        "delft3d-flow_multiple_files_multiple_segments_utm"])
def test_simulations_transect(url):
    response = client.get(url)
    assert response.status_code == 200
    data = response.json()
    assert datetime.strptime(data["time"], "%Y-%m-%dT%H:%M:%S%z")


@pytest.mark.parametrize("url", [
    "/simulations/transect/delft3d-flow/geneva/202304030400/202304050400/46.351,46.294/6.177,6.277",
    "/simulations/transect/delft3d-flow/geneva/202304080400/202304110400/46.351,46.294,46.351/6.177,6.277,6.177",
    "/simulations/transect/delft3d-flow/garda/202312050000/202312150000/45.435,45.589,45.719/10.687,10.635,10.673",
], ids=["delft3d-flow_single_file_single_segment_ch1903_period",
        "delft3d-flow_multiple_files_multiple_segments_ch1903_period",
        "delft3d-flow_multiple_files_multiple_segments_utm_period"])
def test_simulations_transect_period(url):
    response = client.get(url)
    assert response.status_code == 200
    data = response.json()
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")


def test_simulations_depth_time():
    response = client.get("/simulations/depthtime/delft3d-flow/geneva/202304050300/202304112300/46.5/6.67")
    assert response.status_code == 200
    data = response.json()
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")


def test_one_dimensional_simulations_metadata():
    response = client.get("/simulations/1d/metadata")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_one_dimensional_simulations_metadata_lake():
    response = client.get("/simulations/1d/metadata/simstrat/aegeri")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


def test_one_dimensional_simulations_file():
    response = client.get("/simulations/1d/file/simstrat/aegeri/202405")
    assert response.status_code == 200
    content_type = response.headers.get('content-type', '').lower()
    assert 'application/nc' in content_type, "Response content type is not NetCDF"


def test_one_dimensional_simulations_point():
    response = client.get("/simulations/1d/point/simstrat/aegeri/202405050300/202406072300/1")
    assert response.status_code == 200
    data = response.json()
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")
    assert isinstance(data["variables"]["T"]["data"][0], float)

def test_one_dimensional_simulations_profile():
    response = client.get("/simulations/1d/profile/simstrat/aegeri/202405050300")
    assert response.status_code == 200
    data = response.json()
    assert datetime.strptime(data["time"], "%Y-%m-%dT%H:%M:%S%z")
    assert isinstance(data["variables"]["T"]["data"][0], float)


def test_one_dimensional_simulations_depth_time():
    response = client.get("/simulations/1d/depthtime/simstrat/aegeri/202405050300/202406072300")
    assert response.status_code == 200
    data = response.json()
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")
    assert isinstance(data["variables"]["T"]["data"][0][0], float)

def test_one_dimensional_simulations_day_of_year_metadata():
    response = client.get("/simulations/1d/doy/metadata")
    assert response.status_code == 200

def test_one_dimensional_simulations_day_of_year():
    response = client.get("/simulations/1d/doy/simstrat/aegeri/T/1.0")
    assert response.status_code == 200
    data = response.json()
    assert datetime.strptime(data["start_time"], "%Y-%m-%dT%H:%M:%S%z")
    assert datetime.strptime(data["end_time"], "%Y-%m-%dT%H:%M:%S%z")
    assert isinstance(data["variables"]["doy"]["data"][0], int)
    assert isinstance(data["variables"]["mean"]["data"][0], float)
    assert isinstance(data["variables"]["max"]["data"][0], float)
    assert isinstance(data["variables"]["min"]["data"][0], float)
    assert isinstance(data["variables"]["std"]["data"][0], float)

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
    assert isinstance(data, dict)
    assert "time" in data
    assert isinstance(data["time"], list)
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")
    assert "lat" in data
    assert "lng" in data


def test_get_cosmo_area_forecast():
    response = client.get("/meteoswiss/cosmo/area/forecast/VNXZ32/20230101/46.49/6.65/46.51/6.67")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "time" in data
    assert isinstance(data["time"], list)
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")
    assert "lat" in data
    assert "lng" in data


def test_get_cosmo_point_reanalysis():
    response = client.get("/meteoswiss/cosmo/point/reanalysis/VNXQ34/20230101/20230101/46.5/6.67")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "time" in data
    assert isinstance(data["time"], list)
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")
    assert "lat" in data
    assert "lng" in data


def test_get_cosmo_point_forecast():
    response = client.get("/meteoswiss/cosmo/point/forecast/VNXZ32/20230101/46.5/6.67")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "time" in data
    assert isinstance(data["time"], list)
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
    assert isinstance(data, dict)
    assert "time" in data
    assert isinstance(data["time"], list)
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")
    assert "lat" in data
    assert "lng" in data


def test_icon_area_forecast():
    response = client.get("/meteoswiss/icon/area/forecast/icon-ch2-eps/20240703/46.49/6.65/46.51/6.67")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "time" in data
    assert isinstance(data["time"], list)
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")
    assert "lat" in data
    assert "lng" in data


def test_icon_point_reanalysis():
    response = client.get("/meteoswiss/icon/point/reanalysis/kenda-ch1/20240729/20240729/46.49/6.65")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "time" in data
    assert isinstance(data["time"], list)
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")
    assert "lat" in data
    assert "lng" in data


def test_icon_point_forecast():
    response = client.get("/meteoswiss/icon/point/forecast/icon-ch2-eps/20240703/46.49/6.65")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "time" in data
    assert isinstance(data["time"], list)
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")
    assert "lat" in data
    assert "lng" in data


def test_meteoswiss_meteodata_station_metadata_station():
    response = client.get("/meteoswiss/meteodata/metadata/PUY")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_meteoswiss_meteodata_measured():
    parameter = "pva200h0"
    response = client.get("/meteoswiss/meteodata/measured/PUY/{}/20230101/20240210".format(parameter))
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert parameter in data
    assert "time" in data
    assert isinstance(data[parameter], list)
    assert isinstance(data["time"], list)
    assert isinstance(data[parameter][0], float)
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")


def test_bafu_hydrodata_station_metadata():
    response = client.get("/bafu/hydrodata/metadata/2009")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_bafu_hydrodata_measured():
    parameter = "AbflussPneumatikunten"
    response = client.get("/bafu/hydrodata/measured/2009/{}/20210207/20230201".format(parameter))
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert parameter in data
    assert "time" in data
    assert isinstance(data[parameter], list)
    assert isinstance(data["time"], list)
    assert isinstance(data[parameter][0], float)
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")


def test_bafu_hydrodata_measured_resample():
    parameter = "AbflussPneumatikunten"
    response = client.get("/bafu/hydrodata/measured/2009/{}/20210207/20230201?resample=hourly".format(parameter))
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert parameter in data
    assert "time" in data
    assert isinstance(data[parameter], list)
    assert isinstance(data["time"], list)
    assert isinstance(data[parameter][0], float)
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
    assert isinstance(data, dict)
    assert "time" in data
    assert "Secchi depth [m]" in data
    assert isinstance(data["Secchi depth [m]"], list)
    assert isinstance(data["time"], list)
    assert isinstance(data["Secchi depth [m]"][0], float)
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")


@pytest.mark.parametrize("url", [
    "/simulations/transect/delft3d-flow/geneva/202304030400/202304050400/46.351,46.294/6.177,6.277",
    "/simulations/transect/delft3d-flow/geneva/202304080400/202304110400/46.351,46.294,46.351/6.177,6.277,6.177",
    "/simulations/transect/delft3d-flow/garda/202312050000/202312150000/45.435,45.589,45.719/10.687,10.635,10.673",
], ids=["delft3d-flow_single_file_single_segment_ch1903",
        "delft3d-flow_multiple_files_multiple_segments_ch1903",
        "delft3d-flow_multiple_files_multiple_segments_utm"])
def test_simulations_transect_period(url):
    response = client.get(url)
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_one_dimensional_simulations_metadata():
    """
    Test the one_dimensional_simulations_metadata endpoint.
    """
    response = client.get("/simulations/1d/metadata")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_one_dimensional_simulations_metadata_lake():
    """
    Test the one_dimensional_simulations_metadata_lake endpoint.
    """
    response = client.get("/simulations/1d/metadata/simstrat/aegeri")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


def test_one_dimensional_simulations_file():
    """
    Test the one_dimensional_simulations_file endpoint.
    """
    response = client.get("/simulations/1d/file/simstrat/aegeri/202301")
    assert response.status_code == 200
    content_type = response.headers.get('content-type', '').lower()
    assert 'application/nc' in content_type, "Response content type is not NetCDF"


def test_one_dimensional_simulations_point():
    """
    Test the one_dimensional_simulations_point endpoint.
    """
    response = client.get("/simulations/1d/point/simstrat/aegeri/T/202309050300/202309072300/1")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "time" in data
    assert isinstance(data["time"], list)
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")
    assert "T" in data
    assert isinstance(data["T"], list)
    assert isinstance(data["T"][0], float)


def test_one_dimensional_simulations_depth_time():
    """
    Test the one_dimensional_simulations_depth_time endpoint.
    """
    response = client.get("/simulations/1d/depthtime/simstrat/aegeri/T/202309050300/202309072300")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "time" in data
    assert isinstance(data["time"], list)
    assert datetime.strptime(data["time"][0], "%Y-%m-%dT%H:%M:%S%z")
    assert "T" in data
    assert isinstance(data["T"], list)
    assert isinstance(data["T"][0][0], float)


def test_one_dimensional_simulations_day_of_year():
    """
    Test the one_dimensional_simulations_day_of_year endpoint.
    """
    response = client.get("/simulations/1d/doy/simstrat/aegeri/T/1")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "doy" in data
    assert isinstance(data["doy"], list)
    assert isinstance(data["doy"][0], int)
    assert "mean" in data
    assert isinstance(data["mean"], list)
    assert isinstance(data["mean"][0], float)
    assert "max" in data
    assert isinstance(data["max"], list)
    assert isinstance(data["max"][0], float)
    assert "min" in data
    assert isinstance(data["min"], list)
    assert isinstance(data["min"][0], float)
    assert "std" in data
    assert isinstance(data["std"], list)
    assert isinstance(data["std"][0], float)

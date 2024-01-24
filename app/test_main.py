from fastapi.testclient import TestClient
from datetime import datetime
import pytest
from .main import app

client = TestClient(app)


def test_welcome():
    response = client.get("/")
    assert response.status_code == 200


def test_meteoswiss_cosmo_metadata():
    """
    Test the meteoswiss_cosmo_metadata endpoint to ensure it returns the expected list.
    """
    response = client.get("/meteoswiss/cosmo/metadata")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_meteoswiss_meteodata_station_metadata():
    """
    Test the meteoswiss_meteodata_station_metadata endpoint to ensure it returns the expected list.
    """
    response = client.get("/meteoswiss/meteodata/metadata/PUY")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_meteoswiss_meteodata_measured():
    """
    Test the meteoswiss_meteodata_measured endpoint
    """
    parameter = "pva200h0"
    response = client.get("/meteoswiss/meteodata/measured/PUY/{}/19810101/20100101".format(parameter))
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert parameter in data
    assert "Time" in data
    assert isinstance(data[parameter], list)
    assert isinstance(data["Time"], list)
    assert isinstance(data[parameter][0], float)
    assert datetime.strptime(data["Time"][0], "%Y-%m-%dT%H:%M:%S%z")


def test_bafu_hydrodata_station_metadata():
    """
    Test the bafu_hydrodata_station_metadata endpoint
    """
    response = client.get("/bafu/hydrodata/metadata/2009")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_bafu_hydrodata_measured():
    """
    Test the bafu_hydrodata_measured endpoint
    """
    parameter = "AbflussPneumatikunten"
    response = client.get("/bafu/hydrodata/measured/2009/{}/20210207/20230201".format(parameter))
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert parameter in data
    assert "Time" in data
    assert isinstance(data[parameter], list)
    assert isinstance(data["Time"], list)
    assert isinstance(data[parameter][0], float)
    assert datetime.strptime(data["Time"][0], "%Y-%m-%dT%H:%M:%S%z")

def test_bafu_hydrodata_measured_resample():
    """
    Test the bafu_hydrodata_measured endpoint, resampling to one hour
    """
    parameter = "AbflussPneumatikunten"
    response = client.get("/bafu/hydrodata/measured/2009/{}/20210207/20230201?resample=hourly".format(parameter))
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert parameter in data
    assert "Time" in data
    assert isinstance(data[parameter], list)
    assert isinstance(data["Time"], list)
    assert isinstance(data[parameter][0], float)
    assert datetime.strptime(data["Time"][0], "%Y-%m-%dT%H:%M:%S%z")
    assert (datetime.strptime(data["Time"][1], "%Y-%m-%dT%H:%M:%S%z") -
            datetime.strptime(data["Time"][0], "%Y-%m-%dT""%H:%M:%S%z")).total_seconds() == 3600


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

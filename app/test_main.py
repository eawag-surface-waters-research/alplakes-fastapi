from fastapi.testclient import TestClient
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

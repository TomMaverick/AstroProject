# tests/test_milestone17_ztf_fetcher.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from astropy.table import Table

from astro_project.pipeline.ztf_fetcher import query_ztf_archive, download_ztf_batch

@patch("astro_project.pipeline.ztf_fetcher.Irsa.query_region")
def test_query_ztf_archive_filtering_and_sorting(mock_query_region):
    # Create a mock Astropy Table
    mock_table = Table({
        "obsjd": [60000.1, 60000.5, 60000.3, 59999.0, 60000.2],
        "filtercode": ["zr", "zr", "zg", "zr", "zr"],
        "filefracday": ["20230101100000", "20230101500000", "20230101300000", "20221231000000", "20230101200000"],
        "ra": [180.0] * 5,
        "dec": [45.0] * 5,
        "seeing": [1.5, 1.6, 1.4, 2.0, 1.5],
        "fileuri": [
            "http://fake/1.fits", 
            "http://fake/5.fits", 
            "http://fake/3.fits", 
            "http://fake/old.fits", 
            "http://fake/2.fits"
        ]
    })
    mock_query_region.return_value = mock_table

    # Query for MJD 60000.0 to 60000.9, band "r"
    results = query_ztf_archive(
        ra_deg=180.0, 
        dec_deg=45.0, 
        mjd_start=60000.0, 
        mjd_end=60000.9, 
        band="r"
    )

    # Should exclude MJD 59999.0 and band "zg" (60000.3)
    assert len(results) == 3
    
    # Should be sorted by MJD: 60000.1, 60000.2, 60000.5
    assert results[0]["obsjd"] == 60000.1
    assert results[1]["obsjd"] == 60000.2
    assert results[2]["obsjd"] == 60000.5
    
    # Check URL extraction
    assert results[0]["url"] == "http://fake/1.fits"

@patch("astro_project.pipeline.ztf_fetcher.requests.get")
def test_download_ztf_batch(mock_get, tmp_path):
    # Setup mock response
    mock_response = MagicMock()
    mock_response.content = b"fake_fits_data"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    query_results = [
        {"url": "http://fake.url/ztf_image1.fits"},
        {"url": "http://fake.url/ztf_image2.fits"}
    ]

    # Run download
    downloaded_files = download_ztf_batch(query_results, dest_dir=tmp_path)

    assert len(downloaded_files) == 2
    assert mock_get.call_count == 2
    
    # Verify files were written
    assert (tmp_path / "ztf_image1.fits").exists()
    assert (tmp_path / "ztf_image2.fits").exists()

    # Test caching (should not download again)
    mock_get.reset_mock()
    downloaded_files_cached = download_ztf_batch(query_results, dest_dir=tmp_path)
    
    assert len(downloaded_files_cached) == 2
    assert mock_get.call_count == 0  # No new requests made

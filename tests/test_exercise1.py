from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from GKiPO.exercise1 import (
    display_image,
    download_image,
    main,
    print_matrix_table,
    process_image,
)


@pytest.fixture
def dummy_image_file(tmp_path: Path) -> Path:
    """Create a small synthetic dummy image (100x80 BGR) for testing."""
    img_path = tmp_path / "dummy.jpg"
    dummy_img = np.random.randint(0, 256, (80, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), dummy_img)
    return img_path


def test_download_image_already_exists(tmp_path: Path) -> Path:
    """Test that download_image returns immediately if the file already exists."""
    dest = tmp_path / "existing.jpg"
    dest.write_text("fake content")

    result = download_image("http://fake-url.com/img.jpg", dest)
    assert result == dest


@patch("urllib.request.urlopen")
def test_download_image_success(mock_urlopen, tmp_path: Path):
    """Test successful downloading of an image when it doesn't exist locally."""
    dest = tmp_path / "downloaded.jpg"
    mock_response = mock_urlopen.return_value.__enter__.return_value
    mock_response.read.return_value = b"\xff\xd8\xff\xe0\x00\x10JFIF"

    result = download_image("http://fake-url.com/img.jpg", dest)
    assert result == dest
    assert dest.exists()


@patch("urllib.request.urlopen")
def test_download_image_failure(mock_urlopen, tmp_path: Path):
    """Test that network/URLError is properly raised during download failure."""
    import urllib.error

    dest = tmp_path / "fail.jpg"
    mock_urlopen.side_effect = urllib.error.URLError("Network down")

    with pytest.raises(urllib.error.URLError):
        download_image("http://fake-url.com/img.jpg", dest)


def test_process_image_max_dim(dummy_image_file: Path):
    """Test processing with max_dim bounds dimensions."""
    matrix = process_image(dummy_image_file, output_path=None, max_dim=32)
    h, w = matrix.shape
    assert h <= 32
    assert w <= 32
    assert isinstance(matrix, np.ndarray)


def test_process_image_scaling_and_rotation(dummy_image_file: Path, tmp_path: Path):
    """Test 50% scale, greyscale conversion, rotation, and saving to disk."""
    out_path = tmp_path / "processed.jpg"
    matrix = process_image(dummy_image_file, output_path=out_path, max_dim=None)

    assert out_path.exists()
    assert matrix.shape == (50, 40)


def test_process_image_file_not_found():
    """Test that processing a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        process_image(Path("non_existent_image_path.jpg"))


@patch("cv2.imwrite", return_value=False)
def test_process_image_write_failure(
    mock_imwrite, dummy_image_file: Path, tmp_path: Path
):
    """Test that an OSError is raised if cv2.imwrite fails to write the image."""
    out_path = tmp_path / "fail_write.jpg"
    with pytest.raises(OSError, match="Failed to write processed image"):
        process_image(dummy_image_file, output_path=out_path)


def test_print_matrix_table(capsys):
    """Test that matrix table printing runs successfully and outputs table headers."""
    small_matrix = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    print_matrix_table(small_matrix)

    captured = capsys.readouterr()
    assert "Col" in captured.out
    assert "Row  0" in captured.out
    assert " 10" in captured.out


@patch("matplotlib.pyplot.show")
def test_display_image(mock_plt_show, dummy_image_file: Path):
    """Test display_image loads the file and calls plt.show without errors."""
    display_image(dummy_image_file)
    mock_plt_show.assert_called_once()


def test_display_image_file_not_found():
    """Test that display_image raises FileNotFoundError if image doesn't exist."""
    with pytest.raises(FileNotFoundError):
        display_image(Path("missing_display_file.jpg"))


@patch("GKiPO.exercise1.download_image")
@patch("GKiPO.exercise1.process_image")
@patch("GKiPO.exercise1.display_image")
@patch("GKiPO.exercise1.print_matrix_table")
def test_main_success(
    mock_print_table, mock_display, mock_process, mock_download, tmp_path: Path
):
    """Test that main executes all steps successfully when mocked."""
    mock_download.return_value = tmp_path / "raw.jpg"
    mock_process.return_value = np.zeros((10, 10), dtype=np.uint8)

    main()

    mock_download.assert_called_once()
    assert mock_process.call_count == 2
    mock_print_table.assert_called_once()
    mock_display.assert_called_once()


@patch(
    "GKiPO.exercise1.download_image", side_effect=FileNotFoundError("Simulated failure")
)
def test_main_failure(mock_download):
    """Test that main catches exceptions, logs the error, and exits with status 1."""
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1

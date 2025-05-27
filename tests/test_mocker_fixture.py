import pytest

def test_mocker_is_available(mocker):
    mock_obj = mocker.Mock()
    assert mock_obj is not None
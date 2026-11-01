import pytest

def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cpu", help="Device to run tests on (cpu or cuda)")

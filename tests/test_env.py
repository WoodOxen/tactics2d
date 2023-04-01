import sys

sys.path.append(".")
sys.path.append("..")

import pytest
import logging

logging.basicConfig(level=logging.INFO)

from tactics2d.envs import RacingEnv, ParkingEnv


@pytest.mark.skip(reason="not implemented")
def test_parking_env():
    return


@pytest.mark.skip(reason="not implemented")
def test_racing_env():
    return


if __name__ == "__main__":
    env = RacingEnv()

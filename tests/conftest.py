"""Keep CPU checks practical on small laptops and avoid shared RNG surprises."""

import pytest
import torch

from src.utils import seed_everything


@pytest.fixture(autouse=True)
def reproducible_cpu():
    torch.set_num_threads(2)
    seed_everything(42)

"""Specifications from plotting specific cell models."""

import logging

import numpy as np

from collections import (
    namedtuple,
)

from typing import (
    Dict,
)

from ect.specs import Data_spec


logger = logging.getLogger(name=__name__)


def preprocess_wei(
        values: np.ndarray,
        params: Dict[str, float]
) -> Data_spec:
    """Preprocess the Wei cell model for plotting."""
    vol = params["vol"]
    beta0 = params["beta0"]
    voli = values[:, 10]
    volo = (1 + 1/beta0)*vol - voli

    # Rescale the ion concentration by volume to get `mol`.
    values[:, 4] /= volo
    values[:, 5] /= voli
    values[:, 6] /= volo
    values[:, 7] /= voli
    values[:, 8] /= volo
    values[:, 9] /= voli
    values[:, 10] /= volo

    # NB! This relies on tiny dicts being ordered
    variable_dict = {
        "V": (r"Transmembrane potential", "mV"),
        "m": (r"Voltage Gate (m)", "mV"),
        "h": (r"Voltage Gate (n)", "mV"),
        "n": (r"Voltage Gate (h)", "mV"),
        "Ko": (r"Extracellular Potassium $[K^+]$", "mol"),
        "Ki": (r"Intracellular Potessium $[K^+]$", "mol"),
        "Nao": (r"Extracellular Sodium $[Na^+]$", "mol"),
        "Nai": (r"Intracellular Sodium $[Na^+]$", "mol"),
        "Clo": (r"Exctracellular Chlorine $[Cl^-]$", "mol"),
        "Cli": (r"Intracellular Chlorine $[CL^-]$", "mol"),
        "beta": (
            r"Ratio of intracellular to extracellular volume",
            r"$Vol_i/Vol_e$"
        ),
        "O": (r"Extracellular Oxygen $[O_2]$", "mol")
    }

    for i, name in enumerate(("V", "m", "h", "n", "Ko", "Ki", "Nao", "Nai", "Clo", "Cli", "beta", "O")):
        yield Data_spec(values[:, i], name, *variable_dict[name])

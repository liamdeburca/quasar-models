__all__ = [
    "BaseModel",
    "Fitter",
    "InterpolatedSequentialModel",
    "LinearTie",
    "MCMCFitter",
    "PrepareModel",
    "PreparedSequentialModel",
    "SequentialModel",
    "prepare_model",
]

from collections.abc import Generator
from contextlib import contextmanager

from .base_model import BaseModel
from .fitting import Fitter, MCMCFitter
from .interpolated_sequential_model import InterpolatedSequentialModel
from .prepare_model import M, PrepareModel, _CoordsProtocol
from .prepared_sequential_model import PreparedSequentialModel
from .sequential_model import SequentialModel
from .utils import LinearTie


@contextmanager
def prepare_model(
    model: M,
    coords: _CoordsProtocol,
) -> Generator[PreparedSequentialModel, None, None]:
    """
    Model preparation pipeline intended for fitting contexts.
    """
    preparer = PrepareModel.fromCoords(coords, model)
    data = preparer.createDataDict(coords)
    try:
        seq_model, where = preparer.__enter__()
        yield PreparedSequentialModel(
            seq_model=seq_model,
            data=data,
            where=where,
        )
    finally:
        preparer.__exit__(None, None, None)
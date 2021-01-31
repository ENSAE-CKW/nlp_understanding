from pathlib import PurePosixPath
from kedro.io import AbstractDataSet
import pandas as pd
from typing import Dict, Any


class CustomPandas(AbstractDataSet):
    def __init__(self, filepath: str):
        self._filepath = PurePosixPath(filepath)
        pass

    def _load(self) -> pd.DataFrame:
        return pd.read_csv(str(self._filepath))

    def _save(self, data: pd.DataFrame) -> None:
        data.to_csv(str(self._filepath))
        pass

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath)

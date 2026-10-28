from pathlib import Path
from typing import ClassVar

from quasar_typing.pathlib import AnyDirPath, DirPath

STORAGE_DIR: Path = Path.home() / ".quasar_models"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


class PathManager:
    _STORAGE_DIR: ClassVar[DirPath] = STORAGE_DIR

    _BALMER_SERIES_CACHE: ClassVar[AnyDirPath] = STORAGE_DIR / "balmer_series"
    _BALMER_CONTINUUM_CACHE: ClassVar[AnyDirPath] = STORAGE_DIR / "balmer_continuum"
    _HOST_CACHE: ClassVar[AnyDirPath] = STORAGE_DIR / "host"
    _IRON_CACHE: ClassVar[AnyDirPath] = STORAGE_DIR / "iron"

    @classmethod
    def get_storage_dir(cls) -> DirPath:
        cls._STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        return cls._STORAGE_DIR

    @classmethod
    def get_balmer_series_cache(cls) -> DirPath:
        cls._BALMER_SERIES_CACHE.mkdir(parents=True, exist_ok=True)
        return cls._BALMER_SERIES_CACHE

    @classmethod
    def get_balmer_continuum_cache(cls) -> DirPath:
        cls._BALMER_CONTINUUM_CACHE.mkdir(parents=True, exist_ok=True)
        return cls._BALMER_CONTINUUM_CACHE

    @classmethod
    def get_host_cache(cls) -> DirPath:
        cls._HOST_CACHE.mkdir(parents=True, exist_ok=True)
        return cls._HOST_CACHE

    @classmethod
    def get_iron_cache(cls) -> DirPath:
        cls._IRON_CACHE.mkdir(parents=True, exist_ok=True)
        return cls._IRON_CACHE

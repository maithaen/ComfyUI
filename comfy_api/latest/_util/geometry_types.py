from io import BytesIO

import torch


class VOXEL:
    def __init__(self, data: torch.Tensor):
        self.data = data


class MESH:
    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor):
        self.vertices = vertices
        self.faces = faces


class File3D:
    """3D file type storing binary data in memory.

    This is the backing class for all FILE_3D_* ComfyTypes.
    """

    def __init__(self, data: BytesIO, file_format: str):
        self._data = data
        self.format = file_format

    @property
    def data(self) -> BytesIO:
        """Get the BytesIO data, seeking to the beginning."""
        self._data.seek(0)
        return self._data

    def save_to(self, path: str) -> str:
        """Save the 3D file data to disk."""
        self._data.seek(0)
        with open(path, "wb") as f:
            f.write(self._data.read())
        return path

    def __repr__(self) -> str:
        return f"File3D({self.format})"

import numpy as np


class IndexedNullable:
    
    def __init__(self, index, data):
        self.index = index
        self.data = data
    
    def __getitem__(self, item):
        if isinstance(item, int):
            ind = self.index[item]
            return self.data[ind] if ind > 0 else None
        elif isinstance(item, np.ndarray):
            item = np.atleast_1d(item)
            return IndexedNullable(self.index[item], self.data)
        else:
            raise TypeError

    def to_masked(self):
        data = np.empty(len(self.index), dtype=self.data.dtype)
        mask = self.index >= 0
        data[mask] = self.data
        return MaskedNullable(mask, data)


class MaskedNullable:
    
    def __init__(self, mask, data) -> None:
        self.mask = mask
        self.data = data
    
    def __getitem__(self, item):
        if isinstance(item, int):
            m = self.mask[item]
            return self.data[item] if m else None
        elif isinstance(item, np.ndarray):
            item = np.atleast_1d(item)
            return MaskedNullable(self.mask[item], self.data[item])
        else:
            raise TypeError

    def to_indexed(self):
        data = self.data[self.mask]
        index = np.empty(len(data), dtype="int64")
        index[self.mask == 0] = -1
        index[self.mask] = np.arange(len(data))  # could collect uniques
        return IndexedNullable(index, data)


class String:
    def __init__(self, offsets, data) -> None:
        self.offsets = offsets
        self.data = data

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.data[self.offsets[item]: self.offsets[item + 1]].decode()
        else:
            return String(self.offsets.__getitem__(item), self.data)

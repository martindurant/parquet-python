import os

import numpy as np

import fastparquet

data = os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(__file__))
    ),
    "TEST-DATA"
)


def test_short():
    pf = fastparquet.ParquetFile(os.path.join(data, "output_table.parquet"))
    out = pf.to_numpy()
    expected = {
      'foo.with.strings-data': [0, 1, -1],
      'foo.with.strings-cats': ["hey", "there"],
      'foo.with.ints-data': [1, 2, 3],
      'foo.with.lists.list-offsets': [0, 1, 2, 3],
      'foo.with.lists.list.element-data': [0, 0, 0],
      'foo.with.lists.list.element-cats': [0]
    }
    final = {k: list(v) if isinstance(v, np.ndarray) else v
             for k, v in out[0].items()}

    assert final == expected

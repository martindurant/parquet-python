import copy
import os
import pickle

from fastparquet import ParquetFile
from fastparquet.test.util import TEST_DATA

fn = os.path.join(TEST_DATA, "nation.impala.parquet")
pf = ParquetFile(fn)


def test_serialize():
    fmd2 = pickle.loads(pickle.dumps(pf.fmd))
    assert fmd2 == pf.fmd

    rg = pf.row_groups[0]
    rg2 = pickle.loads(pickle.dumps(rg))
    assert rg == rg2


def test_copy():
    fmd2 = copy.copy(pf.fmd)
    assert fmd2 is not pf.fmd
    assert fmd2.row_groups is pf.fmd.row_groups

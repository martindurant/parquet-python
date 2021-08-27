import copy
import os
import pickle

from fastparquet import ParquetFile
from .util import TEST_DATA
from fastparquet.schema import schema_tree

fn = os.path.join(TEST_DATA, "nation.impala.parquet")
pf = ParquetFile(fn)


def test_serialize():
    fmd2 = pickle.loads(pickle.dumps(pf.fmd))
    # we twiddle for what would have been done during ParquetFile instantiation
    for sch in fmd2.schema:
        sch.name = sch.name.decode()
    assert str(fmd2) == str(pf.fmd)

    rg = pf.row_groups[0]
    rg2 = pickle.loads(pickle.dumps(rg))
    assert rg == rg2


def test_copy():
    fmd2 = copy.copy(pf.fmd)
    assert fmd2 is not pf.fmd
    assert fmd2.row_groups == pf.fmd.row_groups

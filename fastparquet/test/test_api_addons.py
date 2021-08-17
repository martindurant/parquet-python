# -*- coding: utf-8 -*-
import pandas as pd
import pytest
from fastparquet import write, ParquetFile
from .util import tempdir


VAL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]
DF = pd.DataFrame({'value' : VAL})
RGO = [0,2,5,8,11]

def test_nth_val_error(tempdir):
    # Test 0 / raise error
    # Test data.
    write(tempdir, DF, row_group_offsets=RGO, file_scheme='hive')
    pf = ParquetFile(tempdir)
    # rg idx :      0    |         1        |       2       |      3      |  4
    # values :  0.1  0.2 | 0.3  *  0.4  0.5 | 0.6  0.7  0.8 | 0.9  1  1.1 | 1.2
    # val    :                 0.31
    n = 12
    val = 0.31
    with pytest.raises(ValueError, match="^13 rows are necessary"):
        res = pf.nth_val('value', n, True, val)

def test_nth_val_1(tempdir):
    # Test 1 / 'val' within a row group.
    # Test data.
    write(tempdir, DF, row_group_offsets=RGO, file_scheme='hive')
    pf = ParquetFile(tempdir)
    # rg idx :      0    |         1        |       2       |      3      |  4
    # values :  0.1  0.2 | 0.3  *  0.4  0.5 | 0.6  0.7  0.8 | 0.9  1  1.1 | 1.2
    # val    :                 0.31
    val = 0.31
    res = pf.nth_val('value', 1, True, val)                 # before=True / n=1
    assert res == (True, 0.3, val)
    res = pf.nth_val('value', 2, True, val)                 # before=True / n=2
    assert res == (True, 0.2, val)
    res = pf.nth_val('value', 2, True, val, exclude=False)  # before=True / n=2
    assert res == (True, 0.2, val)
    res = pf.nth_val('value', 3, True, val)                 # before=True / n=3
    assert res == (True, 0.1, val)
    res = pf.nth_val('value', 1, False, val)               # before=False / n=1
    assert res == (True, val, 0.4)
    res = pf.nth_val('value', 2, False, val)               # before=False / n=2
    assert res == (True, val, 0.5)
    res = pf.nth_val('value', 2, False, val, exclude=False)# before=False / n=2
    assert res == (True, val, 0.5)
    res = pf.nth_val('value', 3, False, val)               # before=False / n=3
    assert res == (True, val, 0.6)
    res = pf.nth_val('value', 4, False, val)               # before=False / n=4
    assert res == (True, val, 0.7)
    res = pf.nth_val('value', 9, False, val)               # before=False / n=9
    assert res == (True, val, 1.2)

    # rg idx :      0    |       1          |       2       |      3      |  4
    # values :  0.1  0.2 | 0.3  0.4  *  0.5 | 0.6  0.7  0.8 | 0.9  1  1.1 | 1.2
    # val    :                      0.41
    val = 0.41
    res = pf.nth_val('value', 2, True, val)                 # before=True / n=2
    assert res == (True, 0.3, val)
    res = pf.nth_val('value', 2, True, val, exclude=False)  # before=True / n=2
    assert res == (True, 0.3, val)
    res = pf.nth_val('value', 2, False, val)               # before=False / n=2
    assert res == (True, val, 0.6)
    res = pf.nth_val('value', 2, False, val, exclude=False)# before=False / n=2
    assert res == (True, val, 0.6)

def test_nth_val_2(tempdir):
    # Test 2 / 'val' on a value.
    # Test data.
    write(tempdir, DF, row_group_offsets=RGO, file_scheme='hive')
    pf = ParquetFile(tempdir)
    # rg idx :        0    |       1       |       2       |      3      |  4
    # values :    0.1  0.2 | 0.3  0.4  0.5 | 0.6  0.7  0.8 | 0.9  1  1.1 | 1.2
    # val    :                    0.4
    val = 0.4
    res = pf.nth_val('value', 2, True, val)                 # before=True / n=2
    assert res == (True, 0.2, val)
    res = pf.nth_val('value', 2, True, val, exclude=False)  # before=True / n=2
    assert res == (True, 0.3, val)
    res = pf.nth_val('value', 2, False, val)               # before=False / n=2
    assert res == (True, val, 0.6)
    res = pf.nth_val('value', 2, False, val, exclude=False)# before=False / n=2
    assert res == (True, val, 0.5)

def test_nth_val_3(tempdir):    
    # Test 3 / 'val' outside.
    # Test data.
    write(tempdir, DF, row_group_offsets=RGO, file_scheme='hive')
    pf = ParquetFile(tempdir)
    # Test 3.0 / before  == True
    # rg idx :     0    |       1       |       2       |      3      |  4
    # values : 0.1  0.2 | 0.3  0.4  0.5 | 0.6  0.7  0.8 | 0.9  1  1.1 | 1.2  *
    # val    :                                                              1.3
    val = 1.3
    res = pf.nth_val('value', 2, True, val)                 # before=True / n=2
    assert res == (True, 1.1, val)
    res = pf.nth_val('value', 2, True, val, exclude=False)  # before=True / n=2
    assert res == (True, 1.1, val)

    # rg idx :         0    |       1       |       2       |      3      |  4
    # values :  *  0.1  0.2 | 0.3  0.4  0.5 | 0.6  0.7  0.8 | 0.9  1  1.1 | 1.2
    # val    :  0
    val = 0
    res = pf.nth_val('value', 2, False, val)               # before=False / n=2
    assert res == (True, val, 0.2)
    res = pf.nth_val('value', 2, False, val, exclude=False)# before=False / n=2
    assert res == (True, val, 0.2)

def test_nth_val_4(tempdir):  
    # Test 4 / 'val' on a bound of a row group.
    # Test data.
    write(tempdir, DF, row_group_offsets=RGO, file_scheme='hive')
    pf = ParquetFile(tempdir)
    # rg idx :        0    |       1       |       2       |      3      |  4
    # values :    0.1  0.2 | 0.3  0.4  0.5 | 0.6  0.7  0.8 | 0.9  1  1.1 | 1.2
    # val    :                                         0.8
    val = 0.8
    res = pf.nth_val('value', 2, True, val)                 # before=True / n=2
    assert res == (True, 0.6, val)
    res = pf.nth_val('value', 2, True, val, exclude=False)  # before=True / n=2
    assert res == (True, 0.7, val)

    # rg idx :        0    |       1       |       2       |      3      |  4
    # values :    0.1  0.2 | 0.3  0.4  0.5 | 0.6  0.7  0.8 | 0.9  1  1.1 | 1.2
    # val    :                               0.6
    val = 0.6
    res = pf.nth_val('value', 2, True, val)                 # before=True / n=2
    assert res == (True, 0.4, val)
    res = pf.nth_val('value', 2, True, val, exclude=False)  # before=True / n=2
    assert res == (True, 0.5, val)

    # rg idx :        0    |       1       |       2       |      3      |  4
    # values :    0.1  0.2 | 0.3  0.4  0.5 | 0.6  0.7  0.8 | 0.9  1  1.1 | 1.2
    # val    :               0.3
    val = 0.3
    res = pf.nth_val('value', 2, False, val)               # before=False / n=2
    assert res == (True, val, 0.5)
    res = pf.nth_val('value', 2, False, val, exclude=False)# before=False / n=2
    assert res == (True, val, 0.4)
    
    # rg idx :        0    |       1       |       2       |      3      |  4
    # values :    0.1  0.2 | 0.3  0.4  0.5 | 0.6  0.7  0.8 | 0.9  1  1.1 | 1.2
    # val    :                         0.5
    val = 0.5
    res = pf.nth_val('value', 2, False, val)               # before=False / n=2
    assert res == (True, val, 0.7)
    res = pf.nth_val('value', 2, False, val, exclude=False)# before=False / n=2
    assert res == (True, val, 0.6)

def test_nth_val_5(tempdir):  
    # Test 5 / 'val' outside, leading to a roll in reversed direction.
    # Test data.
    write(tempdir, DF, row_group_offsets=RGO, file_scheme='hive')
    pf = ParquetFile(tempdir)
    # rg idx :         0    |       1       |       2       |      3      |  4
    # values :  *  0.1  0.2 | 0.3  0.4  0.5 | 0.6  0.7  0.8 | 0.9  1  1.1 | 1.2
    # val    :  0
    val = 0
    res = pf.nth_val('value', 2, True, val)                 # before=True / n=2
    assert res == (False, 0.1, 0.3)
    res = pf.nth_val('value', 2, True, val, exclude=False)  # before=True / n=2
    assert res == (False, 0.1, 0.2)

    # rg idx :     0    |       1       |       2       |      3      |  4
    # values : 0.1  0.2 | 0.3  0.4  0.5 | 0.6  0.7  0.8 | 0.9  1  1.1 | 1.2  *
    # val    :                                                              1.3
    val = 1.3
    res = pf.nth_val('value', 2, False, val)               # before=False / n=2
    assert res == (False, 1.0, 1.2)
    res = pf.nth_val('value', 2, False, val, exclude=False)# before=False / n=2
    assert res == (False, 1.1, 1.2)
    
    # Other out of bound tests.
    # rg idx :      0    |         1        |       2       |      3      |  4
    # values :  0.1  0.2 | 0.3  *  0.4  0.5 | 0.6  0.7  0.8 | 0.9  1  1.1 | 1.2
    # val    :                 0.31
    val = 0.31
    res = pf.nth_val('value', 4, True, val)                # before=False / n=4
    assert res == (False, 0.1, 0.5)
    res = pf.nth_val('value', 10, False, val)             # before=False / n=10
    assert res == (False, 0.2, 1.2)

def test_nth_val_6(tempdir):
    # Test 6 / no 'val'.
    # Test data.
    write(tempdir, DF, row_group_offsets=RGO, file_scheme='hive')
    pf = ParquetFile(tempdir)
    # rg idx :        0    |       1       |       2       |      3      |  4
    # values :    0.1  0.2 | 0.3  0.4  0.5 | 0.6  0.7  0.8 | 0.9  1  1.1 | 1.2
    # val    : None (set by default to 1.2, the last value)
    res = pf.nth_val('value', 2, True)                      # before=True / n=2
    assert res == (True, 1.0, 1.2)
    res = pf.nth_val('value', 2, True, exclude=False)       # before=True / n=2
    assert res == (True, 1.1, 1.2)

    # rg idx :        0    |       1       |       2       |      3      |  4
    # values :    0.1  0.2 | 0.3  0.4  0.5 | 0.6  0.7  0.8 | 0.9  1  1.1 | 1.2
    # val    : None (set by default to 0.1, the 1st value)
    res = pf.nth_val('value', 2, False)                    # before=False / n=2
    assert res == (True, 0.1, 0.3)
    res = pf.nth_val('value', 2, False, exclude=False)     # before=False / n=2
    assert res == (True, 0.1, 0.2)

def test_nth_val_7(tempdir):   
    # Test 7 / 'val' in-between 2 row groups.
    # Test data.
    write(tempdir, DF, row_group_offsets=RGO, file_scheme='hive')
    pf = ParquetFile(tempdir)
    # rg idx :        0    |       1       |       2       |      3      |  4
    # values :    0.1  0.2 | 0.3  0.4  0.5 * 0.6  0.7  0.8 | 0.9  1  1.1 | 1.2
    # val    :                            0.51
    val = 0.51
    res = pf.nth_val('value', 1, True, val)                 # before=True / n=1
    assert res == (True, 0.5, val)
    res = pf.nth_val('value', 2, True, val)                 # before=True / n=2
    assert res == (True, 0.4, val)
    res = pf.nth_val('value', 2, True, val, exclude=False)  # before=True / n=2
    assert res == (True, 0.4, val)
    res = pf.nth_val('value', 5, True, val)                 # before=True / n=5
    assert res == (True, 0.1, val)

    res = pf.nth_val('value', 1, False, val)               # before=False / n=1
    assert res == (True, val, 0.6)
    res = pf.nth_val('value', 2, False, val)               # before=False / n=2
    assert res == (True, val, 0.7)
    res = pf.nth_val('value', 2, False, val, exclude=False)# before=False / n=2
    assert res == (True, val, 0.7)
    res = pf.nth_val('value', 5, False, val)               # before=False / n=5
    assert res == (True, val, 1.0)

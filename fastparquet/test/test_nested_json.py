# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os

import pytest

from fastparquet import json_writer, ParquetFile
from fastparquet.test.util import tempdir

TEST_DATA = "test-data"
_ = tempdir


@pytest.mark.skip()
def test_deep_write(tempdir):

    filename = os.sep.join([tempdir, TEST_DATA])
    data = [
        {"a": [{"b": 1}, {"b": 2}]}
    ]
    json_writer.write(filename, data)

    new_data = ParquetFile(filename).to_pandas().to_dict()



@pytest.mark.skip()
def test_simple_write(tempdir):

    filename = os.sep.join([tempdir, TEST_DATA])
    data = [
        {"a": {"b": 1}}
    ]
    json_writer.write(filename, data)

    new_data = ParquetFile(filename).to_pandas().to_dict()
    assert new_data == data


@pytest.mark.skip()
def test_write_w_nulls(tempdir):

    filename = os.sep.join([tempdir, TEST_DATA])
    data = [
        {"a": {"b": 1}},
        {"a": {"b": None, "c": 2}},
        {"a": None},
        {"a": {"b": 4}}
    ]
    json_writer.write(filename, data)

    new_data = ParquetFile(filename).to_pandas().to_dict()
    assert new_data == data



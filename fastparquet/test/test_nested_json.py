# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os

from fastparquet import ParquetFile, json_writer, json_writer2
from fastparquet.test.util import tempdir

TEST_DATA = "test-data"
_ = tempdir


def test_write(tempdir):

    filename = os.sep.join([tempdir, TEST_DATA])
    data = [
        {"a": [{"b": 1}, {"b": 2}]}
    ]

    json_writer2.write(filename, data)


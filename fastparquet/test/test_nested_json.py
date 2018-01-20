# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os

from fastparquet import json_writer, ParquetFile
from fastparquet.test.util import tempdir

TEST_DATA = "test-data"
_ = tempdir


def test_write(tempdir):

    filename = os.sep.join([tempdir, TEST_DATA])
    data = [
        {"a": [{"b": 1}, {"b": 2}]}
    ]

    json_writer.write(filename, data)

    new_data = ParquetFile(filename).to_pandas().to_dict()
    pass



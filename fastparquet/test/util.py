from itertools import product
import os
import pytest
import tempfile
import shutil

import pandas as pd

TEST_DATA = "test-data"

port = 5555
endpoint_uri = "http://127.0.0.1:%s/" % port


@pytest.fixture()
def s3():
    s3fs = pytest.importorskip("s3fs")
    pytest.importorskip("moto")
    # writable local S3 system
    import shlex
    import subprocess
    import requests
    import time

    if "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = "foo"
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = "foo"
    proc = subprocess.Popen(shlex.split("moto_server s3 -p %s" % port))

    timeout = 5
    while True:
        try:
            r = requests.get(endpoint_uri)
            if r.ok:
                break
        except:
            pass
        timeout -= 0.1
        assert timeout > 0, "Timed out waiting for moto server"
        time.sleep(0.1)
    s3 = s3fs.S3FileSystem(anon=False, client_kwargs={"endpoint_url": endpoint_uri})
    s3.mkdir(TEST_DATA)
    for cat, catnum in product(('fred', 'freda'), ('1', '2', '3')):
        path = os.sep.join([TEST_DATA, 'split', 'cat=' + cat,
                            'catnum=' + catnum])
        files = os.listdir(path)
        for fn in files:
            full_path = os.path.join(path, fn)
            s3.put(full_path, full_path)
    path = os.path.join(TEST_DATA, 'split')
    files = os.listdir(path)
    for fn in files:
        full_path = os.path.join(path, fn)
        if os.path.isdir(full_path):
            continue
        s3.put(full_path, full_path)
    yield s3
    proc.terminate()
    proc.wait()


@pytest.fixture(scope="module")
def sql():
    pyspark = pytest.importorskip("pyspark")
    conf = pyspark.SparkConf()
    conf.set("spark.driver.bindAddress", "127.0.0.1")
    sql = pyspark.sql.SparkSession.builder.appName(
        "Word Count").master("local[*]").config(
        "spark.driver.bindAddress", "127.0.0.1").getOrCreate()
    yield sql
    sql.stop()


@pytest.fixture()
def tempdir():
    d = tempfile.mkdtemp()
    yield d
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)



def makeMixedDataFrame():
    index = pd.Index(["a", "b", "c", "d", "e"], name="index")

    data = {
        "A": pd.Series([0.0, 1.0, 2.0, 3.0, 4.0], dtype="float64"),
        "B": pd.Series([0.0, 1.0, 0.0, 1.0, 0.0], dtype="float64"),
        "C": pd.Series(["foo1", "foo2", "foo3", "foo4", "foo5"], dtype='object'),
        "D": pd.bdate_range("1/1/2009", periods=5),
    }
    return pd.DataFrame(data=data)



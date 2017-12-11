fastparquet/parquet_thrift: fastparquet/parquet.thrift
	thrift -gen py:package_prefix=fastparquet -out $@ $<

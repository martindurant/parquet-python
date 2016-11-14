Usage Notes
===========

Some additional information to bear in mind when using fastparquet,
in no particular order.

Whilst we aim to make the package simple to use, some choices on the part
of the user may effect performance and data consistency.

Categoricals
------------

When writing a data-frame with a column of pandas type ``Category``, the
data will be encoded using parquet "dictionary encoding". This stores all
the possible values of the column (typically strings) separately, and the
index corresponding to each value as a data set of int32 numbers. If there
is a significant performance gain to be made, such as long labels, but low
cardinality, users are suggested to turn their object columns into the
category type:

.. code-block:: python

    df[col] = df[col].astype('category')

When loading, only columns which have dictionary encoding for every row-group,
and are also included in the optional keyword parameter ``categories`` will
result in categorical-type columns in the output. Other columns will be
converted to object type, which is potentially expensive. On the other hand,
if most entries in the column are NULL (see below), then converting to/from
category type is probably unnecessary.

Note that before loading, it is not possible to know whether the above condition
will be met, so the ``dtypes`` attribute of a ``ParquetFile`` will show the
data type appropriate for the values of column and never ``Category``.

Byte Arrays
-----------

Often, information in a column can be encoded in a small number of characters,
perhaps a single character. Given that dictionary encoding (above) requires
four bytes per value, plus additional space and processing to create the
encoding, it can be much more efficient to store the code characters directly.
Conversely, variable-length byte arrays are also slow and inefficient, since
the length of each value needs to be stored.

Fixed-length byte arrays provide the best of both, and will probably be the
most efficient storage where the values are 1-4 characters. They are not,
however, very common in pandas, so the data-type must always be explicitly
given, and converting from an object (variable-length)
column is slightly cumbersome:

.. code-block:: python

    # Create a 1-character length fixed byte column
    data['a'] = np.array([b'a', b'b', b'c', b'd', b'e'], dtype="S1")

    # convert an existing column
    s = data['a'].astype('S3')  # 3-char type
    del data['a']
    data['a'] = s

Furthermore, fixed-length byte arrays are not supported by `spark`, so
files written using this may not be portable.

Nulls
-----

In pandas, NULL values are typically represented by the floating point ``NaN``.
This value can be stored in float and time fields, and will be read back such
that the original data is recovered. They are not, however, the same thing
as missing values, and if querying the resultant files using other frameworks,
this should be born in mind.

Because of the ``NaN`` encoding for NULLs, pandas is unable to represent missing
data in an integer field. In practice, this means that fastparquet will never
write any NULLs in an integer field, and if reading an integer field with NULLs,
the resultant column will become a float type. This is in line with what
pandas does when reading integers.

For object and category columns, NULLs (``None``) do exist, and fastparquet can
read and write them. Including this data does come at a cost, however, and
so we only enable writing the NULLs data if the corresponding column name is
included in the ``has_nulls`` optional keyword. This situation may change in
the future to make writing of the nulls information the default.


Data Types
----------

There is fairly good correspondence between pandas data-types and parquet
simple and logical data types. The `types documentation <https://github.com/Parquet/parquet-format/blob/master/LogicalTypes.md>`_
gives details of the implementation spec.

A couple of caveats should be noted:

- fastparquet will
  not write any Decimal columns, only float, and when reading such columns,
  the output will also be float, with potential machine-precision errors;
- only UTF8 encoding for text is automatically handled, although arbitrary
  byte strings can be written as raw bytes type;
- the time types have millisecond accuracy, whereas pandas time types normally
  are microsecond;
- all times are stored as UTC, and timezone information will
  be lost;
- complex numbers must have their real and imaginary parts stored as two
  separate float columns.

Partitions and row-groups
-------------------------

The parquet format allows for partitioning the data by the values of some
(low-cardinality) columns and by row sequence number. Both of these can be
in operation at the same time, and, in situations where only certain sections
of the data need to be loaded, can produce great performance benefits in
combination with load filters.

Splitting on both row-groups and partitions can potentially result in many
data-files and large metadata. It should be used sparingly, when partial
selecting of the data is anticipated.

**Row groups**

The keyword parameter ``row_group_offsets`` allows control of the row
sequence-wise splits in the data. For example, with the default value,
each row group will contain 50 million rows. The exact index of the start
of each row-group can also be specified, which may be appropriate in the
presence of a monotonic index: such as a time index might lead to the desire
to have all the row-group boundaries coincide with year boundaries in the
data.

**Partitions**

In the presence of some low-cardinality columns, it may be advantageous to
split data data on the values of those columns. This is done by writing a
directory structure with *key=value* names. Multiple partition columns can
be chosen, leading to a multi-level directory tree.

Consider the following directory tree from this `spark example <http://spark.apache.org/docs/latest/sql-programming-guide.html#partition-discovery>`_:

    table/
        gender=male/
           country=US/
              data.parquet
           country=CN/
              data.parquet
        gender=female/
            country=US/
               data.parquet
            country=CN/
               data.parquet

Here the two partitioned fields are *gender* and *country*, each of which have
two possible values, resulting in four datafiles. The corresponding columns
are not stored in the data-files, but inferred on load, so space is saved,
and if selecting based on these values, potentially some of the data need
not be loaded at all.

If there were two row groups and the same partitions as above, each leaf
directory would contain (up to) two files, for a total of eight. If a
row-group happens to contain no data for one of the field value combinations,
that data file is omitted.

Connection to dask
------------------

`dask <http://dask.pydata.org/>`_ provides a pandas-like dataframe interface to
larger-than-memory and distributed datasets, as part of a general parallel
computation engine. In this context, it allows the parallel loading and
processing of the component pieces of a parquet dataset across the cored of
a CPU and/or the nodes of a distributed cluster.

Dask will provide two simple end-user functions:

- ``dask.dataframe.read_parquet`` with keyword options similar to
  ``ParquetFile.to_pandas``. The URL parameter, however, can point to
  various filesystems, such as S3 or HDFS. Loading is *lazy*, only happening
  on demand.
- ``dask.dataframe.DataFrame.to_parquet`` with keyword options similar to
  ``fastparquet.write``. One row-group/file will be generated for each division
  of the dataframe, or, if using partitioning, up to one row-group/file per
  division per partition combination.

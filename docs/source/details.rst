Usage Notes
===========

Some additional information to bear in mind when using fastparquet,
in no particular order. Much of what follows has implications for writing
parquet files that are compatible with other parquet implementations, versus
performance when writing data for reading back with fastparquet.

Whilst we aim to make the package simple to use, some choices on the part
of the user may effect performance and data consistency.

Categoricals
------------

When writing a data-frame with a column of pandas type ``Category``, the
data will be encoded using Parquet "dictionary encoding". This stores all
the possible values of the column (typically strings) separately, and the
index corresponding to each value as a data set of integers. If there
is a significant performance gain to be made, such as long labels, but low
cardinality, users are suggested to turn their object columns into the
category type:

.. code-block:: python

    df[col] = df[col].astype('category')

Fastparquet will automatically use metadata information to load such columns
as categorical *if* the data was written by fastparquet.

To efficiently load a column as a categorical type for data from other
parquet frameworks, include it in the optional
keyword parameter ``categories``; however it must be encoded as dictionary
throughout the dataset.

.. code-block:: python

    pf = ParquetFile('input.parq')
    df = pf.to_pandas(categories={'cat': 12})

Where we provide a hint that the column ``cat`` has up to 12 possible values.
``categories`` can also take a list, in which case up to 32767 (2**15 - 1)
labels are assumed.
Columns that are encoded as dictionary but not included in ``categories`` will
be de-referenced on load which is potentially expensive.

Note that before loading, it is not possible to know whether the above condition
will be met, so the ``dtypes`` attribute of a ``ParquetFile`` will show the
data type appropriate for the values of column, unless the data originates with
fastparquet.

Byte Arrays
-----------

Often, information in a column can be encoded in a small number of characters,
perhaps a single character. Variable-length byte arrays are also slow and
inefficient, however, since the length of each value needs to be stored.

Fixed-length byte arrays provide the best of both, and will probably be the
most efficient storage where the values are 1-4 bytes long, especially if the
cardinality is relatively high for dictionary encoding. To automatically
convert string values to fixed-length when writing, use the ``fixed_text``
optional keyword, with a predetermined length.

.. code-block:: python

    write('out.parq', df, fixed_text={'char_code': 1})

Such an encoding will be the fastest to read, especially if the values are
bytes type, as opposed to UTF8 strings. The values will be converted back
to objects upon loading.

Fixed-length byte arrays are not supported by Spark, so
files written using this may not be portable.

Short-type Integers
-------------------

Types like 1-byte ints (signed or unsigned) are stored using bitpacking for
optimized space and speed. Unfortunately, Spark is known not to be
able to handle these types. If you want to generate files for reading by
Spark, be sure to transform integer columns to a minimum of 4 bytes (numpy
``int32`` or ``uint32``) before saving.

Nulls
-----

In pandas, there is no internal representation difference between NULL (no value)
and NaN (not a valid number) for float, time and category columns. Whether to
enocde these values using parquet NULL or the "sentinel" values is a choice for
the user. The parquet framework that will read the data will likely treat
NULL and NaN differently (e.g., in `in Spark`_). In the typical case of tabular
data (as opposed to strict numerics), users often mean the NULL semantics, and
so should write NULLs information. Furthermore, it is typical for some parquet
frameworks to define all columns as optional, whether or not they are intended to
hold any missing data, to allow for possible mutation of the schema when appending
partitions later.

.. _in Spark: https://spark.apache.org/docs/2.1.0/sql-programming-guide.html#nan-semantics

Since there is some cost associated with reading and writing NULLs information,
fastparquet provides the ``has_nulls`` keyword when writing to specify how to
handle NULLs. In the case that a column has no NULLs, including NULLs information
will not produce a great performance hit on reading, and only a slight extra time
upon writing, while determining that there are zero NULL values.

The following cases are allowed for ``has_nulls``:

    - True: all columns become optional, and NaNs are always stored as NULL. This is
      the best option for compatibility. This is the default.

    - False: all columns become required, and any NaNs are stored as NaN; if there
      are any fields which cannot store such sentinel values (e.g,. string),
      but do contain None, there will be an error.

    - 'infer': only object columns will become optional, since float, time, and
      category columns can store sentinel values, and pandas int columns cannot
      contain any NaNs. This is the best-performing
      option if the data will only be read by fastparquet.

    - list of strings: the named columns will be optional, others required (no NULLs)

This value can be stored in float and time fields, and will be read back such
that the original data is recovered. They are not, however, the same thing
as missing values, and if querying the resultant files using other frameworks,
this should be born in mind. With ``has_nulls=None`` (the default) on writing,
float, time and category fields will not write separate NULLs information, and
the metadata will give num_nulls=0.


Data Types
----------

There is fairly good correspondence between pandas data-types and Parquet
simple and logical data types. The `types documentation <https://github.com/apache/parquet-format/blob/master/LogicalTypes.md>`_
gives details of the implementation spec.

A couple of caveats should be noted:

- fastparquet will
  not write any Decimal columns, only float, and when reading such columns,
  the output will also be float, with potential machine-precision errors;
- only UTF8 encoding for text is automatically handled, although arbitrary
  byte strings can be written as raw bytes type;
- the time types have microsecond accuracy, whereas pandas time types normally
  are nanosecond accuracy;
- all times are stored as UTC, and timezone information will
  be lost;
- complex numbers must have their real and imaginary parts stored as two
  separate float columns.

Spark Timestamps
----------------

Fastparquet can read and write int96-style timestamps, as typically found in Apache
Spark and Map-Reduce output.

Currently, int96-style timestamps are the only known use of the int96 type without
an explicit schema-level converted type assignment. They will be automatically converted to
times upon loading.

Similarly on writing, the ``times`` keyword controls the encoding of timestamp columns:
"int64" is the default and faster option, producing parquet standard compliant data, but
"int96" is required to write data which is compatible with Spark.

Reading Nested Schema
---------------------

Fastparquet can read nested schemas. The principal mechamism is *flattening*, whereby
parquet schema struct columns become top-level columns. For instance, if a schema looks
like

.. code-block:: python

    root
    | - visitor: OPTIONAL
      | - ip: BYTE_ARRAY, UTF8, OPTIONAL
        - network_id: BYTE_ARRAY, UTF8, OPTIONAL

then the ``ParquetFile`` will include entries "visitor.ip" and "visitor.network_id" in its
``columns``, and these will become ordinary Pandas columns.

Fastparquet also handles some parquet LIST and MAP types. For instance, the schema may include

.. code-block:: python

    | - tags: LIST, OPTIONAL
        - list: REPEATED
           - element: BYTE_ARRAY, UTF8, OPTIONAL

In this case, ``columns`` would include an entry "tags", which evaluates to an object column
containing lists of strings. Reading such columns will be relatively slow.
If the 'element' type is anything other than a primitive type,
i.e., a struct, map or list, than fastparquet will not be able to read it, and the resulting
column will either not be contained in the output, or contain only ``None`` values.

Partitions and row-groups
-------------------------

The Parquet format allows for partitioning the data by the values of some
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

Consider the following directory tree from this `Spark example <http://Spark.apache.org/docs/latest/sql-programming-guide.html#partition-discovery>`_:

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


Iteration
---------

For data-sets too big to fit conveniently into memory, it is possible to
iterate through the row-groups in a similar way to reading by chunks from
CSV with pandas.

.. code-block:: python

    pf = ParquetFile('myfile.parq')
    for df in pf.iter_row_groups():
        print(df.shape)
        # process sub-data-frame df

Thus only one row-group is in memory at a time. The same set of options
are available as in ``to_pandas`` allowing, for instance, reading only
specific columns, loading to
categoricals or to ignore some row-groups using filtering.

To get the first row-group only, one would go:

.. code-block:: python

    first = next(iter(pf.iter_row_groups()))

Connection to Dask
------------------

Dask usage is still in development. Expect the features to lag behind
those in fastparquet, and sometimes to become incompatible, if a change has
been made in the one but not the other.

`Dask <http://dask.pydata.org/>`_ provides a pandas-like dataframe interface to
larger-than-memory and distributed datasets, as part of a general parallel
computation engine. In this context, it allows the parallel loading and
processing of the component pieces of a Parquet dataset across the cored of
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

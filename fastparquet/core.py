import io
import logging
import numpy as np

from fastparquet import encoding
from fastparquet.encoding import read_plain, DECODE_TYPEMAP
import fastparquet.cencoding as encoding
from fastparquet.compression import decompress_data, rev_map, decom_into
from fastparquet.converted_types import convert, simple, converts_inplace
from fastparquet.speedups import unpack_byte_array
from fastparquet import parquet_thrift
from fastparquet.cencoding import ThriftObject
from fastparquet.util import val_to_num, simple_concat
from fastparquet.schema import SchemaHelper

logger = logging.getLogger("fastparquet.core")


def _read_page(file_obj, page_header, column_metadata):
    """Read the data page from the given file-object and convert it to raw,
    uncompressed bytes (if necessary)."""
    raw_bytes = file_obj.read(page_header.compressed_page_size)
    raw_bytes = decompress_data(
        raw_bytes,
        page_header.uncompressed_page_size,
        column_metadata.codec,
    )

    if column_metadata.codec:
        assert len(raw_bytes) == page_header.uncompressed_page_size, \
            "found {0} raw bytes (expected {1})".format(
                len(raw_bytes),
                page_header.uncompressed_page_size)
    return raw_bytes


def read_data(fobj, coding, count, bit_width, out=None):
    """For definition and repetition levels

    Reads with RLE/bitpacked hybrid, where length is given by first byte.

    out: potentially provide a len(count) uint8 array to reuse
    """
    out = np.empty(count, dtype=np.uint8) if out is None else out
    o = encoding.NumpyIO(out)
    if coding == parquet_thrift.Encoding.RLE:
        while o.tell() < count:
            encoding.read_rle_bit_packed_hybrid(fobj, bit_width, 0, o, itemsize=1)
    else:
        raise NotImplementedError('Encoding %s' % coding)
    return out


def read_def(io_obj, daph, helper, metadata, out=None):
    """
    Read the definition levels from this page, if any.
    """
    definition_levels = None
    num_nulls = 0
    if not helper.is_required(metadata.path_in_schema):
        max_definition_level = helper.max_definition_level(
            metadata.path_in_schema)
        bit_width = encoding.width_from_max_int(max_definition_level)
        if bit_width:
            # NB: num_values is index 1 for either type of page header
            definition_levels = read_data(
                    io_obj, parquet_thrift.Encoding.RLE,
                    daph.num_values, bit_width, out=out)
        if False and (
                daph.statistics is not None
                and getattr(daph.statistics, "null_count", None) is not None
        ):
            num_nulls = daph.statistics.null_count
        elif False and (
                daph.num_values == metadata.num_values
                and metadata.statistics
                and getattr(metadata.statistics, "null_count", None) is not None
        ):
            num_nulls = metadata.statistics.null_count
        else:
            num_nulls = daph.num_values - (definition_levels ==
                                               max_definition_level).sum()
        if num_nulls == 0:
            definition_levels = None
    return definition_levels, num_nulls


def read_rep(io_obj, daph, helper, metadata, out=None):
    """
    Read the repetition levels from this page, if any.
    """
    repetition_levels = None
    if len(metadata.path_in_schema) > 1:
        max_repetition_level = helper.max_repetition_level(
            metadata.path_in_schema)
        if max_repetition_level == 0:
            repetition_levels = None
        else:
            bit_width = encoding.width_from_max_int(max_repetition_level)
            # NB: num_values is index 1 for either type of page header
            repetition_levels = read_data(io_obj, parquet_thrift.Encoding.RLE,
                                          daph.num_values,
                                          bit_width,
                                          out=out)
    return repetition_levels


def read_data_page(f, helper, header, metadata, skip_nulls=False,
                   selfmade=False):
    """Read a data page: definitions, repetitions, values (in order)

    Only values are guaranteed to exist, e.g., for a top-level, required
    field.
    """
    daph = header.data_page_header
    raw_bytes = _read_page(f, header, metadata)
    io_obj = encoding.NumpyIO(raw_bytes)

    repetition_levels = read_rep(io_obj, daph, helper, metadata)

    if skip_nulls and not helper.is_required(metadata.path_in_schema):
        num_nulls = 0
        definition_levels = None
        skip_definition_bytes(io_obj, daph.num_values)
    else:
        definition_levels, num_nulls = read_def(io_obj, daph, helper, metadata)

    nval = daph.num_values - num_nulls
    se = helper.schema_element(metadata.path_in_schema)
    if daph.encoding == parquet_thrift.Encoding.PLAIN:
        width = se.type_length
        values = read_plain(io_obj.read(),
                            metadata.type,
                            int(daph.num_values - num_nulls),
                            width=width,
                            utf=se.converted_type == 0)
    elif daph.encoding in [parquet_thrift.Encoding.PLAIN_DICTIONARY,
                           parquet_thrift.Encoding.RLE_DICTIONARY,
                           parquet_thrift.Encoding.RLE]:
        # bit_width is stored as single byte.
        if metadata.type == parquet_thrift.Type.BOOLEAN:
            bit_width = 1
        elif daph.encoding == parquet_thrift.Encoding.RLE:
            bit_width = se.type_length
        else:
            bit_width = io_obj.read_byte()
        if bit_width in [8, 16, 32] and selfmade:
            num = (encoding.read_unsigned_var_int(io_obj) >> 1) * 8
            values = np.frombuffer(io_obj.read(num * bit_width // 8),
                                   dtype='int%i' % bit_width)
        elif bit_width:
            if bit_width > 8:
                values = np.empty(daph.num_values-num_nulls, dtype=np.int32)
                o = encoding.NumpyIO(values.view('uint8'))
                encoding.read_rle_bit_packed_hybrid(
                            io_obj, bit_width, io_obj.len-io_obj.tell(), o=o, itemsize=4)
            else:
                values = np.empty(daph.num_values-num_nulls, dtype=np.uint8)
                o = encoding.NumpyIO(values)
                encoding.read_rle_bit_packed_hybrid(
                    io_obj, bit_width, io_obj.len-io_obj.tell(), o=o, itemsize=1)
            if isinstance(values, np.ndarray):
                values = values[:nval]
            else:
                values = values.data[:nval]
        else:
            values = np.zeros(nval, dtype=np.int8)
    elif daph.encoding == parquet_thrift.Encoding.DELTA_BINARY_PACKED:
        values = np.empty(daph.num_values - num_nulls,
                          dtype=np.int64 if metadata.type == 2 else np.int32)
        o = encoding.NumpyIO(values.view('uint8'))
        encoding.delta_binary_unpack(io_obj, o, longval=metadata.type == 2)
    else:
        raise NotImplementedError('Encoding %s' % daph.encoding)
    return definition_levels, repetition_levels, values[:nval]


def skip_definition_bytes(io_obj, num):
    # for self-made set of valid values; this could be cython but is rarely called
    io_obj.seek(6, 1)
    n = num // 64
    while n:
        io_obj.seek(1, 1)
        n //= 128


def read_dictionary_page(file_obj, schema_helper, page_header, column_metadata, utf=False):
    """Read a page containing dictionary data.

    Consumes data using the plain encoding and returns an array of values.
    """
    raw_bytes = _read_page(file_obj, page_header, column_metadata)
    if column_metadata.type == parquet_thrift.Type.BYTE_ARRAY:
        values = unpack_byte_array(
            raw_bytes, page_header.dictionary_page_header.num_values, utf=utf)
    else:
        width = schema_helper.schema_element(
            column_metadata.path_in_schema).type_length
        values = read_plain(
                raw_bytes, column_metadata.type,
                page_header.dictionary_page_header.num_values, width)
    return values


def read_data_page_v2(infile, schema_helper, se, data_header2, cmd,
                      dic, assign, num, use_cat, file_offset, ph, idx=None,
                      selfmade=False, row_filter=None):
    """
    :param infile: open file
    :param schema_helper:
    :param se: schema element
    :param data_header2: page header struct
    :param cmd: column metadata
    :param dic: any dictionary labels encountered
    :param assign: output array (all of it)
    :param num: offset, rows so far
    :param use_cat: output is categorical?
    :return: None

    test data "/Users/mdurant/Downloads/datapage_v2.snappy.parquet"
          a  b    c      d          e
    0   abc  1  2.0   True  [1, 2, 3]
    1   abc  2  3.0   True       None
    2   abc  3  4.0   True       None
    3  None  4  5.0  False  [1, 2, 3]
    4   abc  5  2.0   True     [1, 2]

    b is delta encoded; c is dict encoded

    """
    if data_header2.encoding not in [parquet_thrift.Encoding.PLAIN_DICTIONARY,
                                     parquet_thrift.Encoding.RLE_DICTIONARY,
                                     parquet_thrift.Encoding.RLE,
                                     parquet_thrift.Encoding.PLAIN,
                                     parquet_thrift.Encoding.DELTA_BINARY_PACKED
                                     ]:
        raise NotImplementedError
    size = (ph.compressed_page_size - data_header2.repetition_levels_byte_length -
            data_header2.definition_levels_byte_length)
    data = infile.tell() + data_header2.definition_levels_byte_length + data_header2.repetition_levels_byte_length
    n_values = data_header2.num_values - data_header2.num_nulls

    max_rep = schema_helper.max_repetition_level(cmd.path_in_schema)
    if max_rep:
        # TODO: probably not functional
        bit_width = encoding.width_from_max_int(max_rep)
        io_obj = encoding.NumpyIO(infile.read(data_header2.repetition_levels_byte_length))
        repi = np.empty(data_header2.num_values, dtype="uint8")
        encoding.read_rle_bit_packed_hybrid(io_obj, bit_width, data_header2.num_values,
                                            encoding.NumpyIO(repi), itemsize=1)

    max_def = schema_helper.max_definition_level(cmd.path_in_schema)

    nullable = False
    if max_def and data_header2.num_nulls:
        bit_width = encoding.width_from_max_int(max_def)
        # not the same as read_data(), because we know the length
        io_obj = encoding.NumpyIO(infile.read(data_header2.definition_levels_byte_length))
        if nullable:
            defi = assign._mask
        else:
            # TODO: in tabular data, nulls arrays could be reused for each column
            defi = np.empty(data_header2.num_values, dtype=np.uint8)
        encoding.read_rle_bit_packed_hybrid(io_obj, bit_width, data_header2.num_values,
                                            encoding.NumpyIO(defi), itemsize=1)
        if max_rep:
            # assemble_objects needs both arrays
            nulls = defi != max_def
        else:
            np.not_equal(defi.view("uint8"), max_def, out=defi)
            nulls = defi.view(np.bool_)
    infile.seek(data)

    # input and output element sizes match
    see = se.type_length == assign.dtype.itemsize * 8 or simple.get(se.type).itemsize == assign.dtype.itemsize
    # can read-into
    into0 = ((use_cat or converts_inplace(se) and see)
             and data_header2.num_nulls == 0
             and max_rep == 0 and assign.dtype.kind != "O" and row_filter is None
             and assign.dtype.kind not in "Mm")  # TODO: this can be done in place but is complex
    if row_filter is None:
        row_filter = Ellipsis
    # can decompress-into
    if data_header2.is_compressed is None:
        data_header2.is_compressed = True
    into = (data_header2.is_compressed and rev_map[cmd.codec] in decom_into
            and into0)
    if nullable:
        assign = assign._data

    uncompressed_page_size = (ph.uncompressed_page_size - data_header2.definition_levels_byte_length -
                              data_header2.repetition_levels_byte_length)
    if into0 and data_header2.encoding == parquet_thrift.Encoding.PLAIN and (
            not data_header2.is_compressed or cmd.codec == parquet_thrift.CompressionCodec.UNCOMPRESSED
    ):
        # PLAIN read directly into output (a copy for remote files)
        assign[num:num+n_values].view('uint8')[:] = infile.read(size)
        convert(assign[num:num+n_values], se)
    elif into and data_header2.encoding == parquet_thrift.Encoding.PLAIN:
        # PLAIN decompress directly into output
        decomp = decom_into[rev_map[cmd.codec]]
        decomp(np.frombuffer(infile.read(size), dtype="uint8"),
               assign[num:num+data_header2.num_values].view('uint8'))
        convert(assign[num:num+n_values], se)
    elif data_header2.encoding == parquet_thrift.Encoding.PLAIN:
        # PLAIN, but with nulls or not in-place conversion
        codec = cmd.codec if data_header2.is_compressed else "UNCOMPRESSED"
        raw_bytes = decompress_data(np.frombuffer(infile.read(size), "uint8"),
                                    uncompressed_page_size, codec)
        values = read_plain(raw_bytes,
                            cmd.type,
                            n_values,
                            width=se.type_length,
                            utf=se.converted_type == 0)
        if data_header2.num_nulls:
            if nullable:
                assign[num:num+data_header2.num_values][~nulls[row_filter]] = convert(values, se)[row_filter]
            else:
                assign[num:num+data_header2.num_values][nulls[row_filter]] = None  # or nan or nat
                if row_filter is Ellipsis:
                    assign[num:num+data_header2.num_values][~nulls] = convert(values, se)
                else:
                    assign[num:num+data_header2.num_values][~nulls[row_filter]] = convert(values, se)[row_filter[~nulls]]
        else:
            assign[num:num+data_header2.num_values] = convert(values, se)[row_filter]
    elif (use_cat and data_header2.encoding in [
        parquet_thrift.Encoding.PLAIN_DICTIONARY,
        parquet_thrift.Encoding.RLE_DICTIONARY,
    ]) or (data_header2.encoding == parquet_thrift.Encoding.RLE):
        # DICTIONARY or BOOL direct decode RLE into output (no nulls)
        codec = cmd.codec if data_header2.is_compressed else "UNCOMPRESSED"
        raw_bytes = np.frombuffer(infile.read(size), dtype='uint8')
        raw_bytes = decompress_data(raw_bytes, uncompressed_page_size, codec)
        pagefile = encoding.NumpyIO(raw_bytes)
        if data_header2.encoding != parquet_thrift.Encoding.RLE:
            # TODO: check this bit; is the varint read only row byte-exact fastpath?
            bit_width = pagefile.read_byte()
            encoding.read_unsigned_var_int(pagefile)
        else:
            bit_width = 1
            pagefile.seek(4, 1)
        if bit_width in [8, 16, 32] and selfmade:
            # special fastpath for cats
            outbytes = raw_bytes[pagefile.tell():]
            if len(outbytes) == assign[num:num+data_header2.num_values].nbytes:
                assign[num:num+data_header2.num_values].view('uint8')[row_filter] = outbytes[row_filter]
            else:
                if data_header2.num_nulls == 0:
                    assign[num:num+data_header2.num_values][row_filter] = outbytes[row_filter]
                else:
                    if row_filter is Ellipsis:
                        assign[num:num+data_header2.num_values][~nulls] = outbytes
                    else:
                        assign[num:num + data_header2.num_values][~nulls[row_filter]] = outbytes[~nulls * row_filter]
                    assign[num:num+data_header2.num_values][nulls[row_filter]] = -1
        else:
            if data_header2.num_nulls == 0:
                encoding.read_rle_bit_packed_hybrid(
                    pagefile,
                    bit_width,
                    uncompressed_page_size,
                    encoding.NumpyIO(assign[num:num+data_header2.num_values].view('uint8')),
                    itemsize=bit_width
                )
            else:
                temp = np.empty(data_header2.num_values, assign.dtype)
                encoding.read_rle_bit_packed_hybrid(
                    pagefile,
                    bit_width,
                    uncompressed_page_size,
                    encoding.NumpyIO(temp.view('uint8')),
                    itemsize=bit_width
                )
                if not nullable:
                    assign[num:num+data_header2.num_values][nulls[row_filter]] = None
                assign[num:num+data_header2.num_values][~nulls[row_filter]] = temp[row_filter]

    elif data_header2.encoding in [
        parquet_thrift.Encoding.PLAIN_DICTIONARY,
        parquet_thrift.Encoding.RLE_DICTIONARY
    ]:
        # DICTIONARY to be de-referenced, with or without nulls
        codec = cmd.codec if data_header2.is_compressed else "UNCOMPRESSED"
        compressed_bytes = np.frombuffer(infile.read(size), "uint8")
        raw_bytes = decompress_data(compressed_bytes, uncompressed_page_size, codec)
        out = np.empty(n_values, dtype='uint32')
        pagefile = encoding.NumpyIO(raw_bytes)
        bit_width = pagefile.read_byte()
        encoding.read_rle_bit_packed_hybrid(
            pagefile,
            bit_width,
            uncompressed_page_size,
            encoding.NumpyIO(out.view("uint8")),
            itemsize=4
        )
        if max_rep:
            # num_rows got filled, but consumed num_values data entries
            encoding._assemble_objects(
                assign[idx[0]:idx[0]+data_header2.num_rows], defi, repi, out, dic, d=True,
                null=True, null_val=False, max_defi=max_def, prev_i=0
            )
            idx[0] += data_header2.num_rows
        elif data_header2.num_nulls:
            if not nullable and assign.dtype != "O":
                assign[num:num+data_header2.num_values][nulls] = None  # may be unnecessary
            assign[num:num+data_header2.num_values][~nulls[row_filter]] = dic[out][row_filter]
        else:
            assign[num:num+data_header2.num_values][row_filter] = dic[out][row_filter]
    elif data_header2.encoding == parquet_thrift.Encoding.DELTA_BINARY_PACKED:
        assert data_header2.num_nulls == 0, "null delta-int not implemented"
        codec = cmd.codec if data_header2.is_compressed else "UNCOMPRESSED"
        raw_bytes = decompress_data(np.frombuffer(infile.read(size), "uint8"),
                                    uncompressed_page_size, codec)
        if converts_inplace(se):
            encoding.delta_binary_unpack(
                encoding.NumpyIO(raw_bytes),
                encoding.NumpyIO(assign[num:num+data_header2.num_values].view('uint8'))
            )
            convert(assign[num:num+data_header2.num_values], se)
        else:
            out = np.empty(data_header2.num_values, dtype='int32')
            encoding.delta_binary_unpack(
                encoding.NumpyIO(raw_bytes), encoding.NumpyIO(out.view('uint8'))
            )
            assign[num:num+data_header2.num_values][row_filter] = convert(out, se)[row_filter]
    else:
        raise NotImplementedError
    return data_header2.num_values


# TODO: this executor should not persist between runs to free up threads.
import concurrent.futures
ex = concurrent.futures.ThreadPoolExecutor()


def _run(raw: bytes, ph: ThriftObject, dph: ThriftObject, o: int,
         cmd, rep_width, reps, def_width, defs, assign, colname, se,
         with_data=True):
    io = encoding.NumpyIO(decompress_data(raw, ph.uncompressed_page_size, cmd.codec))
    if rep_width:
        read_data(
            io, 
            parquet_thrift.Encoding.RLE, 
            dph.num_values,
            rep_width,
            out=reps[o:o + dph.num_values]
        )
    if def_width:
        read_data( 
            io, 
            parquet_thrift.Encoding.RLE, 
            dph.num_values,
            def_width,
            out=defs[o:o + dph.num_values]
        )
    if with_data:
        dtype = DECODE_TYPEMAP[se.type]
        # PLAIN - testing
        o = np.frombuffer(io.read(), dtype=dtype)
        assign[f"{colname}-data"].append(o)

DUMMY = np.zeros(1, dtype="int64")


def read_col(column: ThriftObject, schema_helper: SchemaHelper, infile: io.IOBase, 
             assign: dict, use_cat: bool = False, read_data=True):
    """Using the given metadata, read one column in one row-group.

    Parameters
    ----------
    column: thrift structure
        Details on the column
    schema_helper: schema.SchemaHelper
        Based on the schema for this parquet data
    infile: open file or string
        If a string, will open; if an open object, will use as-is
    use_cat: bool (False)
        If this column is encoded throughout with dict encoding, give back
        a pandas categorical column; otherwise, decode to values
    read_data: bool (True)
        Whether to ingest the payload values. If False, only reads offset/index
        information, if they are not already in the output.
    """
    cmd = column.meta_data
    colname = ".".join(cmd.path_in_schema)
    se = schema_helper.schema_element(cmd.path_in_schema)
    rows = cmd.num_values

    # make outputs
    OPT = parquet_thrift.FieldRepetitionType.OPTIONAL
    REP = parquet_thrift.FieldRepetitionType.REPEATED
    rep_map = np.zeros(256, dtype="uint8")  # which rep value maps to which offset arr
    rep_flags = np.zeros(256, dtype="uint8")  # is array offset or index?
    parts = []  # schema names as we walk
    offsets = []  # list of output offset/index arrays
    nreps = 0
    i = 0
    for part in cmd.path_in_schema:
        parts.append(part)
        if schema_helper.schema_element(parts).repetition_type == OPT:
            name = f'{".".join(parts)}-index'
            if name not in assign:
                o = np.empty(rows, dtype="int64")
                assign[name] = o
                offsets.append(o)
                rep_flags[i] = 1
            else:
                offsets.append(DUMMY)
                rep_flags[i] = 2
            i += 1
        elif schema_helper.schema_element(parts).repetition_type == REP:
            # offset has one extra element for closing last list
            name = f'{".".join(parts)}-offsets'
            if name not in assign:
                o = np.empty(rows + 1, dtype="int64")
                assign[name] = o
                offsets.append(o)
            else:
                offsets.append(DUMMY)
                rep_flags[i] = 2  # do not use
            nreps += 1
            i += 1
            rep_map[nreps] = i
    
    # read bytes from source    
    off = min((cmd.dictionary_page_offset or cmd.data_page_offset,
               cmd.data_page_offset))
    infile.seek(off)
    column_binary = infile.read(cmd.total_compressed_size)
    infile = encoding.NumpyIO(column_binary)

    # ingest page headers
    num = 0  # how far through the output we are
    dic = None
    parts = []  # data values from each page
    headers = []  # header objects
    raws = []  # bytes
    assign[f"{colname}-data"] = []

    while num < rows:
        # TODO: if read_data is False, and we have offsets already (OPT 4), can skip all
        ph = ThriftObject.from_buffer(infile, "PageHeader")
        raw = infile.read(ph.compressed_page_size)
        isdict = ph.type == parquet_thrift.PageType.DICTIONARY_PAGE
        headers.append(ph)
        raws.append(raw)
        if isdict and use_cat and f"{colname}-dict" in assign:
            # assert that the dict inherited from elsewhere is fine
            continue
        if ph.type == parquet_thrift.PageType.DATA_PAGE:
            num += ph.data_page_header.num_values
        elif ph.type == parquet_thrift.PageType.DATA_PAGE_V2:
            # NB: this one always knows its num_nulls, but v1 does not
            num += ph.data_page_header_v2.num_values
        elif ph.type == parquet_thrift.PageType.DICTIONARY_PAGE:
            assert dic is None
            dic = True  # should always be first page
        else:
            raise NotImplementedError
    logger.debug("Column %s,  %s headers (inc dict page %s)", colname, len(headers),
                 bool(dic))

    max_rep = schema_helper.max_repetition_level(cmd.path_in_schema)
    max_def = schema_helper.max_definition_level(cmd.path_in_schema)
    
    # Set optimisation conditions, so that we can avoid unecessary reads
    if all(_ is DUMMY for _ in offsets):
        # no deps/defs needed
        if read_data is False:
            return
        OPT = 4
    elif rep_flags.max() == 0:
        OPT = 1
    elif all(_ is DUMMY for _ in offsets[:-1]) and rep_flags[i - 1] == 1:
        OPT = 2
    elif (rep_flags[:max_def] == 1).all():
        OPT = 3
    else:
        OPT = 0

    # process raw pages
    reps = np.empty(cmd.num_values, dtype="uint8")
    defs = np.empty(cmd.num_values, dtype="uint8")
    rep_width = encoding.width_from_max_int(max_rep)
    def_width = encoding.width_from_max_int(max_def)
    futs = []
    off = 0  # counts number of rep/defs across pages
    for i, (raw, ph) in enumerate(zip(raws, headers)):
        dph = ph.data_page_header
        # CONDITIONS:
        #  - (1) need all reps/defs
        #  - (2) need data
        #  - (3) only need last defs (optional, but offsets known)
        # if (1) or (2) or (3)  and v1, must decompress and read sequentially
        # if (1) and v2, need to read reps and defs
        # if (2) and v2, need to decompress and read data
        # if (3) and v2, need to read defs (not reps)
        l = lambda raw=raw, ph=ph, dph=dph, off=off: _run(
                    raw, ph, dph, off, cmd, rep_width, reps, 
                    def_width, defs, assign, colname, se
                )
        if i + 1 < len(raws):
            futs.append(ex.submit(l))
        else:
            l()
        off += dph.num_values
    if futs:
        concurrent.futures.wait(futs)

    assign[f"{colname}-data"] = simple_concat(*assign[f"{colname}-data"])

    # Optimization conditions
    if OPT == 4:
        return
    ocounts = np.zeros(len(offsets) + 1, dtype="uint64")
    if OPT == 1:
        encoding.make_offsets_and_masks_no_nulls(reps, defs, offsets, ocounts)
    elif OPT == 2:
        count = encoding.one_level_optional(defs, offsets[-1], 0, max_def)
    elif OPT == 3:
        encoding.make_offsets_and_masks_no_reps(defs, offsets, ocounts)
    else:
        encoding.make_offsets_and_masks(reps, defs, offsets, rep_map, rep_flags, ocounts)
    for o, count, flag in zip(offsets, ocounts, rep_flags):
        if o is not DUMMY:
            o.resize(count + flag, refcheck=False)


def read_row_group_arrays(file, rg, columns, categories, schema_helper, cats,
                          selfmade=False, assign=None, row_filter=False):
    """
    Read a row group and return as a dict of arrays

    Note that categorical columns (if appearing in the parameter categories)
    will be pandas Categorical objects: the codes and the category labels
    are arrays.
    """
    for column in rg.columns:

        name = ".".join(column.meta_data.path_in_schema)
        if name not in columns:
            continue
        read_col(column, schema_helper, file, use_cat=False,
                 assign=assign)


def read_row_group(file, rg, columns, categories, schema_helper, cats,
                   selfmade=False, index=None, assign=None,
                   scheme='hive', partition_meta=None, row_filter=False):
    """
    Access row-group in a file and read some columns into a data-frame.
    """
    partition_meta = partition_meta or {}
    read_row_group_arrays(file, rg, columns, categories, schema_helper,
                          cats, selfmade, assign=assign, row_filter=row_filter)

    for cat in cats:
        if cat not in assign:
            # do no need to have partition columns in output
            continue
        if scheme == 'hive':
            partitions = [s.split("=") for s in rg.columns[0].file_path.split("/")]
        else:
            partitions = [('dir%i' % i, v) for (i, v) in enumerate(
                rg.columns[0].file_path.split('/')[:-1])]
        key, val = [p for p in partitions if p[0] == cat][0]
        val = val_to_num(val, meta=partition_meta.get(key))
        assign[cat][:] = cats[cat].index(val)

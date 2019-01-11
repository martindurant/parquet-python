# encoding: utf-8
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
from __future__ import absolute_import, division, unicode_literals

import gzip
from io import BytesIO
import struct
from tempfile import TemporaryFile
import time
import zipfile
import zlib

from mo_future import PY3, long, text_type
from mo_logs import Log
from mo_logs.exceptions import suppress_exception
import mo_math

# LIBRARY TO DEAL WITH BIG DATA ARRAYS AS ITERATORS OVER (IR)REGULAR SIZED
# BLOCKS, OR AS ITERATORS OVER LINES

DEBUG = False
MIN_READ_SIZE = 8 * 1024
MAX_STRING_SIZE = 1 * 1024 * 1024


class FileString(text_type):
    """
    ACTS LIKE A STRING, BUT IS A FILE
    """

    def __init__(self, file):
        self.file = file

    def decode(self, encoding):
        if encoding != "utf8":
            Log.error("can not handle {{encoding}}",  encoding= encoding)
        self.encoding = encoding
        return self

    def split(self, sep):
        if sep != "\n":
            Log.error("Can only split by lines")
        self.file.seek(0)
        return LazyLines(self.file)

    def __len__(self):
        temp = self.file.tell()
        self.file.seek(0, 2)
        file_length = self.file.tell()
        self.file.seek(temp)
        return file_length

    def __getslice__(self, i, j):
        j = mo_math.min(j, len(self))
        if j - 1 > 2 ** 28:
            Log.error("Slice of {{num}} bytes is too big", num=j - i)
        try:
            self.file.seek(i)
            output = self.file.read(j - i).decode(self.encoding)
            return output
        except Exception as e:
            Log.error(
                "Can not read file slice at {{index}}, with encoding {{encoding}}",
                index=i,
                encoding=self.encoding,
                cause=e
            )

    def __add__(self, other):
        self.file.seek(0, 2)
        self.file.write(other)

    def __radd__(self, other):
        new_file = TemporaryFile()
        new_file.write(other)
        self.file.seek(0)
        for l in self.file:
            new_file.write(l)
        new_file.seek(0)
        return FileString(new_file)

    def __getattr__(self, attr):
        return getattr(self.file, attr)

    def __del__(self):
        self.file, temp = None, self.file
        if temp:
            temp.close()

    def __iter__(self):
        self.file.seek(0)
        return self.file

    if PY3:
        def __str__(self):
            if self.encoding == "utf8":
                temp = self.file.tell()
                self.file.seek(0, 2)
                file_length = self.file.tell()
                self.file.seek(0)
                output = self.file.read(file_length).decode(self.encoding)
                self.file.seek(temp)
                return output
    else:
        def __unicode__(self):
            if self.encoding == "utf8":
                temp = self.file.tell()
                self.file.seek(0, 2)
                file_length = self.file.tell()
                self.file.seek(0)
                output = self.file.read(file_length).decode(self.encoding)
                self.file.seek(temp)
                return output


def safe_size(source):
    """
    READ THE source UP TO SOME LIMIT, THEN COPY TO A FILE IF TOO BIG
    RETURN A str() OR A FileString()
    """

    if source is None:
        return None

    total_bytes = 0
    bytes = []
    b = source.read(MIN_READ_SIZE)
    while b:
        total_bytes += len(b)
        bytes.append(b)
        if total_bytes > MAX_STRING_SIZE:
            try:
                data = FileString(TemporaryFile())
                for bb in bytes:
                    data.write(bb)
                del bytes
                del bb
                b = source.read(MIN_READ_SIZE)
                while b:
                    total_bytes += len(b)
                    data.write(b)
                    b = source.read(MIN_READ_SIZE)
                data.seek(0)
                Log.note("Using file of size {{length}} instead of str()",  length= total_bytes)

                return data
            except Exception as e:
                Log.error("Could not write file > {{num}} bytes",  num= total_bytes, cause=e)
        b = source.read(MIN_READ_SIZE)

    data = b"".join(bytes)
    del bytes
    return data


class LazyLines(object):
    """
    SIMPLE LINE ITERATOR, BUT WITH A BIT OF CACHING TO LOOK LIKE AN ARRAY
    """

    def __init__(self, source, encoding="utf8"):
        """
        ASSUME source IS A LINE ITERATOR OVER utf8 ENCODED BYTE STREAM
        """
        self.source = source
        self.encoding = encoding
        self._iter = self.__iter__()
        self._last = None
        self._next = 0

    def __getslice__(self, i, j):
        if i == self._next - 1:
            def output():
                yield self._last
                for v in self._iter:
                    self._next += 1
                    yield v

            return output()
        if i == self._next:
            return self._iter
        Log.error("Do not know how to slice this generator")

    def __iter__(self):
        def output():
            for v in self.source:
                self._last = v
                yield self._last

        return output()

    def __getitem__(self, item):
        try:
            if item == self._next:
                self._next += 1
                return self._iter.next()
            elif item == self._next - 1:
                return self._last
            else:
                Log.error("can not index out-of-order too much")
        except Exception as e:
            Log.error("Problem indexing", e)


class CompressedLines(LazyLines):
    """
    KEEP COMPRESSED HTTP (content-type: gzip) IN BYTES ARRAY
    WHILE PULLING OUT ONE LINE AT A TIME FOR PROCESSING
    """

    def __init__(self, compressed, encoding="utf8"):
        """
        USED compressed BYTES TO DELIVER LINES OF TEXT
        LIKE LazyLines, BUT HAS POTENTIAL TO seek()
        """
        self.compressed = compressed
        LazyLines.__init__(self, None, encoding=encoding)
        self._iter = self.__iter__()

    def __iter__(self):
        return LazyLines(ibytes2ilines(compressed_bytes2ibytes(self.compressed, MIN_READ_SIZE), encoding=self.encoding)).__iter__()

    def __getslice__(self, i, j):
        if i == self._next:
            return self._iter

        if i == 0:
            return self.__iter__()

        if i == self._next - 1:
            def output():
                yield self._last
                for v in self._iter:
                    yield v

            return output()
        Log.error("Do not know how to slice this generator")

    def __getitem__(self, item):
        try:
            if item == self._next:
                self._last = self._iter.next()
                self._next += 1
                return self._last
            elif item == self._next - 1:
                return self._last
            else:
                Log.error("can not index out-of-order too much")
        except Exception as e:
            Log.error("Problem indexing", e)


    def __radd__(self, other):
        new_file = TemporaryFile()
        new_file.write(other)
        self.file.seek(0)
        for l in self.file:
            new_file.write(l)
        new_file.seek(0)
        return FileString(new_file)


def compressed_bytes2ibytes(compressed, size):
    """
    CONVERT AN ARRAY OF BYTES TO A BYTE-BLOCK GENERATOR
    USEFUL IN THE CASE WHEN WE WANT TO LIMIT HOW MUCH WE FEED ANOTHER
    GENERATOR (LIKE A DECOMPRESSOR)
    """
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    for i in range(0, mo_math.ceiling(len(compressed), size), size):
        try:
            block = compressed[i: i + size]
            yield decompressor.decompress(block)
        except Exception as e:
            Log.error("Not expected", e)


def ibytes2ilines(generator, encoding="utf8", flexible=False, closer=None):
    """
    CONVERT A GENERATOR OF (ARBITRARY-SIZED) byte BLOCKS
    TO A LINE (CR-DELIMITED) GENERATOR

    :param generator:
    :param encoding: None TO DO NO DECODING
    :param closer: OPTIONAL FUNCTION TO RUN WHEN DONE ITERATING
    :return:
    """
    decode = get_decoder(encoding=encoding, flexible=flexible)
    _buffer = generator.next()
    s = 0
    e = _buffer.find(b"\n")
    while True:
        while e == -1:
            try:
                next_block = generator.next()
                _buffer = _buffer[s:] + next_block
                s = 0
                e = _buffer.find(b"\n")
            except StopIteration:
                _buffer = _buffer[s:]
                del generator
                if closer:
                    closer()
                if _buffer:
                    yield decode(_buffer)
                return

        yield decode(_buffer[s:e])
        s = e + 1
        e = _buffer.find(b"\n", s)


def ibytes2icompressed(source):
    yield (
        b'\037\213\010\000' +  # Gzip file, deflate, no filename
        struct.pack('<L', long(time.time())) +  # compression start time
        b'\002\377'  # maximum compression, no OS specified
    )

    crc = zlib.crc32(b"")
    length = 0
    compressor = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS, zlib.DEF_MEM_LEVEL, 0)
    for d in source:
        crc = zlib.crc32(d, crc) & 0xffffffff
        length += len(d)
        chunk = compressor.compress(d)
        if chunk:
            yield chunk
    yield compressor.flush()
    yield struct.pack("<2L", crc, length & 0xffffffff)


class GzipLines(CompressedLines):
    """
    SAME AS CompressedLines, BUT USING THE GzipFile FORMAT FOR COMPRESSED BYTES
    """

    def __init__(self, compressed, encoding="utf8"):
        CompressedLines.__init__(self, compressed, encoding=encoding)

    def __iter__(self):
        buff = BytesIO(self.compressed)
        return LazyLines(gzip.GzipFile(fileobj=buff, mode='r'), encoding=self.encoding).__iter__()


class ZipfileLines(CompressedLines):
    """
    SAME AS CompressedLines, BUT USING THE ZipFile FORMAT FOR COMPRESSED BYTES
    """

    def __init__(self, compressed, encoding="utf8"):
        CompressedLines.__init__(self, compressed, encoding=encoding)

    def __iter__(self):
        buff = BytesIO(self.compressed)
        archive = zipfile.ZipFile(buff, mode='r')
        names = archive.namelist()
        if len(names) != 1:
            Log.error("*.zip file has {{num}} files, expecting only one.",  num= len(names))
        stream = archive.open(names[0], "r")
        return LazyLines(sbytes2ilines(stream), encoding=self.encoding).__iter__()


def icompressed2ibytes(source):
    """
    :param source: GENERATOR OF COMPRESSED BYTES
    :return: GENERATOR OF BYTES
    """
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    last_bytes_count = 0  # Track the last byte count, so we do not show too many debug lines
    bytes_count = 0
    for bytes_ in source:
        try:
            data = decompressor.decompress(bytes_)
        except Exception as e:
            Log.error("problem", cause=e)
        bytes_count += len(data)
        if mo_math.floor(last_bytes_count, 1000000) != mo_math.floor(bytes_count, 1000000):
            last_bytes_count = bytes_count
            DEBUG and Log.note("bytes={{bytes}}", bytes=bytes_count)
        yield data


def scompressed2ibytes(stream):
    """
    :param stream:  SOMETHING WITH read() METHOD TO GET MORE BYTES
    :return: GENERATOR OF UNCOMPRESSED BYTES
    """
    def more():
        try:
            while True:
                bytes_ = stream.read(4096)
                if not bytes_:
                    return
                yield bytes_
        except Exception as e:
            Log.error("Problem iterating through stream", cause=e)
        finally:
            with suppress_exception:
                stream.close()

    return icompressed2ibytes(more())


def sbytes2ilines(stream, encoding="utf8", closer=None):
    """
    CONVERT A STREAM (with read() method) OF (ARBITRARY-SIZED) byte BLOCKS
    TO A LINE (CR-DELIMITED) GENERATOR
    """
    def read():
        try:
            while True:
                bytes_ = stream.read(4096)
                if not bytes_:
                    return
                yield bytes_
        except Exception as e:
            Log.error("Problem iterating through stream", cause=e)
        finally:
            try:
                stream.close()
            except Exception:
                pass

            if closer:
                try:
                    closer()
                except Exception:
                    pass

    return ibytes2ilines(read(), encoding=encoding)


def get_decoder(encoding, flexible=False):
    """
    RETURN FUNCTION TO PERFORM DECODE
    :param encoding: STRING OF THE ENCODING
    :param flexible: True IF YOU WISH TO TRY OUR BEST, AND KEEP GOING
    :return: FUNCTION
    """
    if encoding == None:
        def no_decode(v):
            return v
        return no_decode
    elif flexible:
        def do_decode1(v):
            return v.decode(encoding, 'ignore')
        return do_decode1
    else:
        def do_decode2(v):
            return v.decode(encoding)
        return do_decode2

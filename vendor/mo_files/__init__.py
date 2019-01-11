# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
import base64
from datetime import datetime
import io
from mimetypes import MimeTypes
import os
import re
import shutil
from tempfile import NamedTemporaryFile, mkdtemp

from mo_dots import Null, coalesce, get_module, is_list
from mo_files.url import URL
from mo_future import PY3, binary_type, text_type, is_text
from mo_logs import Except, Log
from mo_logs.exceptions import extract_stack
from mo_threads import Thread, Till

mime = MimeTypes()


class File(object):
    """
    ASSUMES ALL FILE CONTENT IS UTF8 ENCODED STRINGS
    """

    def __new__(cls, filename, buffering=2 ** 14, suffix=None):
        if filename == None:
            return Null
        elif isinstance(filename, File):
            return filename
        else:
            return object.__new__(cls)

    def __init__(self, filename, buffering=2 ** 14, suffix=None, mime_type=None):
        """
        YOU MAY SET filename TO {"path":p, "key":k} FOR CRYPTO FILES
        """
        if isinstance(filename, File):
            return

        self._mime_type = mime_type

        if isinstance(filename, (binary_type, text_type)):
            try:
                self.key = None
                if filename==".":
                    self._filename = ""
                elif filename.startswith("~"):
                    home_path = os.path.expanduser("~")
                    if os.sep == "\\":
                        home_path = home_path.replace(os.sep, "/")
                    if home_path.endswith("/"):
                        home_path = home_path[:-1]
                    filename = home_path + filename[1::]
                self._filename = filename.replace(os.sep, "/")  # USE UNIX STANDARD
            except Exception as e:
                Log.error(u"can not load {{file}}", file=filename, cause=e)
        else:
            try:
                self.key = base642bytearray(filename.key)
                self._filename = "/".join(filename.path.split(os.sep))  # USE UNIX STANDARD
            except Exception as e:
                Log.error(u"can not load {{file}}", file=filename.path, cause=e)

        while self._filename.find(".../") >= 0:
            # LET ... REFER TO GRANDPARENT, .... REFER TO GREAT-GRAND-PARENT, etc...
            self._filename = self._filename.replace(".../", "../../")
        self.buffering = buffering

        if suffix:
            self._filename = File.add_suffix(self._filename, suffix)

    @classmethod
    def new_instance(cls, *path):
        return File(join_path(*path))

    def __div__(self, other):
        return File(join_path(self, other))

    def __truediv__(self, other):
        return File(join_path(self, other))

    def __rtruediv__(self, other):
        return File(join_path(other, self))

    @property
    def timestamp(self):
        output = os.path.getmtime(self.abspath)
        return output

    @property
    def filename(self):
        return self._filename.replace("/", os.sep)

    @property
    def abspath(self):
        if self._filename.startswith("~"):
            home_path = os.path.expanduser("~")
            if os.sep == "\\":
                home_path = home_path.replace(os.sep, "/")
            if home_path.endswith("/"):
                home_path = home_path[:-1]

            return home_path + self._filename[1::]
        else:
            if os.sep == "\\":
                return os.path.abspath(self._filename).replace(os.sep, "/")
            else:
                return os.path.abspath(self._filename)

    @staticmethod
    def add_suffix(filename, suffix):
        """
        ADD suffix TO THE filename (NOT INCLUDING THE FILE EXTENSION)
        """
        path = filename.split("/")
        parts = path[-1].split(".")
        i = max(len(parts) - 2, 0)
        parts[i] = parts[i] + suffix
        path[-1] = ".".join(parts)
        return "/".join(path)

    @property
    def extension(self):
        parts = self._filename.split("/")[-1].split(".")
        if len(parts) == 1:
            return ""
        else:
            return parts[-1]

    @property
    def name(self):
        parts = self.abspath.split("/")[-1].split(".")
        if len(parts) == 1:
            return parts[0]
        else:
            return ".".join(parts[0:-1])

    @property
    def mime_type(self):
        if not self._mime_type:
            if self.abspath.endswith(".js"):
                self._mime_type = "application/javascript"
            elif self.abspath.endswith(".css"):
                self._mime_type = "text/css"
            elif self.abspath.endswith(".json"):
                self._mime_type = "application/json"
            else:
                self._mime_type, _ = mime.guess_type(self.abspath)
                if not self._mime_type:
                    self._mime_type = "application/binary"
        return self._mime_type

    def find(self, pattern):
        """
        :param pattern: REGULAR EXPRESSION TO MATCH NAME (NOT INCLUDING PATH)
        :return: LIST OF File OBJECTS THAT HAVE MATCHING NAME
        """
        output = []

        def _find(dir):
            if re.match(pattern, dir._filename.split("/")[-1]):
                output.append(dir)
            if dir.is_directory():
                for c in dir.children:
                    _find(c)
        _find(self)
        return output

    def set_extension(self, ext):
        """
        RETURN NEW FILE WITH GIVEN EXTENSION
        """
        path = self._filename.split("/")
        parts = path[-1].split(".")
        if len(parts) == 1:
            parts.append(ext)
        else:
            parts[-1] = ext

        path[-1] = ".".join(parts)
        return File("/".join(path))

    def set_name(self, name):
        """
        RETURN NEW FILE WITH GIVEN EXTENSION
        """
        path = self._filename.split("/")
        parts = path[-1].split(".")
        if len(parts) == 1:
            path[-1] = name
        else:
            path[-1] = name + "." + parts[-1]
        return File("/".join(path))

    def backup_name(self, timestamp=None):
        """
        RETURN A FILENAME THAT CAN SERVE AS A BACKUP FOR THIS FILE
        """
        suffix = datetime2string(coalesce(timestamp, datetime.now()), "%Y%m%d_%H%M%S")
        return File.add_suffix(self._filename, suffix)

    def read(self, encoding="utf8"):
        """
        :param encoding:
        :return:
        """
        with open(self._filename, "rb") as f:
            if self.key:
                return get_module("mo_math.crypto").decrypt(f.read(), self.key)
            else:
                content = f.read().decode(encoding)
                return content

    def read_zipfile(self, encoding='utf8'):
        """
        READ FIRST FILE IN ZIP FILE
        :param encoding:
        :return: STRING
        """
        from zipfile import ZipFile
        with ZipFile(self.abspath) as zipped:
            for num, zip_name in enumerate(zipped.namelist()):
                return zipped.open(zip_name).read().decode(encoding)


    def read_lines(self, encoding="utf8"):
        with open(self._filename, "rb") as f:
            for line in f:
                yield line.decode(encoding).rstrip()

    def read_json(self, encoding="utf8", flexible=True, leaves=True):
        content = self.read(encoding=encoding)
        value = get_module(u"mo_json").json2value(content, flexible=flexible, leaves=leaves)
        abspath = self.abspath
        if os.sep == "\\":
            abspath = "/" + abspath.replace(os.sep, "/")
        return get_module("mo_json_config").expand(value, "file://" + abspath)

    def is_directory(self):
        return os.path.isdir(self._filename)

    def read_bytes(self):
        try:
            if not self.parent.exists:
                self.parent.create()
            with open(self._filename, "rb") as f:
                if self.key:
                    return get_module("mo_math.crypto").decrypt(f.read(), self.key)
                else:
                    return f.read()
        except Exception as e:
            Log.error(u"Problem reading file {{filename}}", filename=self.abspath, cause=e)

    def write_bytes(self, content):
        if not self.parent.exists:
            self.parent.create()
        with open(self._filename, "wb") as f:
            if self.key:
                f.write(get_module("mo_math.crypto").encrypt(content, self.key))
            else:
                f.write(content)

    def write(self, data):
        if not self.parent.exists:
            self.parent.create()
        with open(self._filename, "wb") as f:
            if is_list(data) and self.key:
                Log.error(u"list of data and keys are not supported, encrypt before sending to file")

            if is_list(data):
                pass
            elif isinstance(data, (binary_type, text_type)):
                data=[data]
            elif hasattr(data, "__iter__"):
                pass

            for d in data:
                if not is_text(d):
                    Log.error(u"Expecting unicode data only")
                if self.key:
                    from mo_math.crypto import encrypt
                    f.write(encrypt(d, self.key).encode("utf8"))
                else:
                    f.write(d.encode("utf8"))

    def __iter__(self):
        # NOT SURE HOW TO MAXIMIZE FILE READ SPEED
        # http://stackoverflow.com/questions/8009882/how-to-read-large-file-line-by-line-in-python
        # http://effbot.org/zone/wide-finder.htm
        def output():
            try:
                path = self._filename
                if path.startswith("~"):
                    home_path = os.path.expanduser("~")
                    path = home_path + path[1::]

                with io.open(path, "rb") as f:
                    for line in f:
                        yield line.decode('utf8').rstrip()
            except Exception as e:
                Log.error(u"Can not read line from {{filename}}", filename=self._filename, cause=e)

        return output()

    def append(self, content, encoding='utf8'):
        """
        add a line to file
        """
        if not self.parent.exists:
            self.parent.create()
        with open(self._filename, "ab") as output_file:
            if not is_text(content):
                Log.error(u"expecting to write unicode only")
            output_file.write(content.encode(encoding))
            output_file.write(b"\n")

    def __len__(self):
        return os.path.getsize(self.abspath)

    def add(self, content):
        return self.append(content)

    def extend(self, content):
        try:
            if not self.parent.exists:
                self.parent.create()
            with open(self._filename, "ab") as output_file:
                for c in content:
                    if isinstance(c, str):
                        Log.error(u"expecting to write unicode only")

                    output_file.write(c.encode("utf8"))
                    output_file.write(b"\n")
        except Exception as e:
            Log.error(u"Could not write to file", e)

    def delete(self):
        try:
            if os.path.isdir(self._filename):
                shutil.rmtree(self._filename)
            elif os.path.isfile(self._filename):
                os.remove(self._filename)
            return self
        except Exception as e:
            e = Except.wrap(e)
            if u"The system cannot find the path specified" in e:
                return
            Log.error(u"Could not remove file", e)

    def backup(self):
        path = self._filename.split("/")
        names = path[-1].split(".")
        if len(names) == 1 or names[0] == '':
            backup = File(self._filename + ".backup " + datetime.utcnow().strftime("%Y%m%d %H%M%S"))
        else:
            backup = File.new_instance(
                "/".join(path[:-1]),
                ".".join(names[:-1]) + ".backup " + datetime.now().strftime("%Y%m%d %H%M%S") + "." + names[-1]
            )
        File.copy(self, backup)
        return backup

    def create(self):
        try:
            os.makedirs(self._filename)
        except Exception as e:
            Log.error(u"Could not make directory {{dir_name}}",  dir_name= self._filename, cause=e)

    @property
    def children(self):
        return [File(self._filename + "/" + c) for c in os.listdir(self.filename)]

    @property
    def leaves(self):
        for c in os.listdir(self.abspath):
            child = File(self._filename + "/" + c)
            if child.is_directory():
                for l in child.leaves:
                    yield l
            else:
                yield child

    @property
    def parent(self):
        if not self._filename or self._filename==".":
            return File("..")
        elif self._filename.endswith(".."):
            return File(self._filename+"/..")
        else:
            return File("/".join(self._filename.split("/")[:-1]))

    @property
    def exists(self):
        if self._filename in ["", "."]:
            return True
        try:
            return os.path.exists(self._filename)
        except Exception as e:
            return False

    def __bool__(self):
        return self.__nonzero__()

    def __nonzero__(self):
        """
        USED FOR FILE EXISTENCE TESTING
        """
        if self._filename in ["", "."]:
            return True
        try:
            return os.path.exists(self._filename)
        except Exception as e:
            return False

    @classmethod
    def copy(cls, from_, to_):
        _copy(File(from_), File(to_))

    def __data__(self):
        return self._filename

    def __unicode__(self):
        return self.abspath

    def __str__(self):
        return self.abspath


class TempDirectory(File):
    """
    A CONTEXT MANAGER FOR AN ALLOCATED, BUT UNOPENED TEMPORARY DIRECTORY
    WILL BE DELETED WHEN EXITED
    """
    def __new__(cls):
        return object.__new__(cls)

    def __init__(self):
        File.__init__(self, mkdtemp())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Thread.run("delete dir " + self.name, delete_daemon, file=self, caller_stack=extract_stack(1))


class TempFile(File):
    """
    A CONTEXT MANAGER FOR AN ALLOCATED, BUT UNOPENED TEMPORARY FILE
    WILL BE DELETED WHEN EXITED
    """
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, filename=None):
        if isinstance(filename, File):
            return
        self.temp = NamedTemporaryFile(delete=False)
        self.temp.close()
        File.__init__(self, self.temp.name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Thread.run("delete file " + self.name, delete_daemon, file=self, caller_stack=extract_stack(1))


def _copy(from_, to_):
    if from_.is_directory():
        for c in os.listdir(from_.abspath):
            _copy(from_ / c, to_ / c)
    else:
        File.new_instance(to_).write_bytes(File.new_instance(from_).read_bytes())


if PY3:
    def base642bytearray(value):
        if value == None:
            return bytearray(b"")
        else:
            return bytearray(base64.b64decode(value))
else:
    def base642bytearray(value):
        if value == None:
            return bytearray(b"")
        else:
            return bytearray(base64.b64decode(value))


def datetime2string(value, format="%Y-%m-%d %H:%M:%S"):
    try:
        return value.strftime(format)
    except Exception as e:
        Log.error(u"Can not format {{value}} with {{format}}", value=value, format=format, cause=e)



def join_path(*path):
    def scrub(i, p):
        p = p.replace(os.sep, "/")
        if p in ('', '/'):
            return "."
        if p[-1] == '/':
            p = p[:-1]
        if i > 0 and p[0] == '/':
            p = p[1:]
        return p

    path = [p._filename if isinstance(p, File) else p for p in path]
    abs_prefix = ''
    if path and path[0]:
        if path[0][0] == '/':
            abs_prefix = '/'
            path[0] = path[0][1:]
        elif os.sep == '\\' and path[0][1:].startswith(':/'):
            # If windows, then look for the "c:/" prefix
            abs_prefix = path[0][0:3]
            path[0] = path[0][3:]

    scrubbed = []
    for i, p in enumerate(path):
        scrubbed.extend(scrub(i, p).split("/"))
    simpler = []
    for s in scrubbed:
        if s == ".":
            pass
        elif s == "..":
            if simpler:
                if simpler[-1] == '..':
                    simpler.append(s)
                else:
                    simpler.pop()
            elif abs_prefix:
                raise Exception("can not get parent of root")
            else:
                simpler.append(s)
        else:
            simpler.append(s)

    if not simpler:
        if abs_prefix:
            joined = abs_prefix
        else:
            joined = "."
    else:
        joined = abs_prefix + ('/'.join(simpler))

    return joined


def delete_daemon(file, caller_stack, please_stop):
    # WINDOWS WILL HANG ONTO A FILE FOR A BIT AFTER WE CLOSED IT
    while not please_stop:
        try:
            file.delete()
            return
        except Exception as e:
            e = Except.wrap(e)
            e.trace = e.trace[0:2]+caller_stack

            Log.warning(u"problem deleting file {{file}}", file=file.abspath, cause=e)
            (Till(seconds=10)|please_stop).wait()

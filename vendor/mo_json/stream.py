# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import json
from collections import Mapping
from types import GeneratorType

from mo_dots import split_field, startswith_field, relative_field, Data, join_field, Null, wrap
from mo_logs import Log

DEBUG = False

MIN_READ_SIZE = 8*1024
WHITESPACE = b" \n\r\t"
CLOSE = {
    b"{": b"}",
    b"[": b"]"
}
NO_VARS = set()

json_decoder = json.JSONDecoder().decode



def parse(json, query_path, expected_vars=NO_VARS):
    """
    INTENDED TO TREAT JSON AS A STREAM; USING MINIMAL MEMORY WHILE IT ITERATES
    THROUGH THE STRUCTURE.  ASSUMING THE JSON IS LARGE, AND HAS A HIGH LEVEL
    ARRAY STRUCTURE, IT WILL yield EACH OBJECT IN THAT ARRAY.  NESTED ARRAYS
    ARE HANDLED BY REPEATING THE PARENT PROPERTIES FOR EACH MEMBER OF THE
    NESTED ARRAY. DEEPER NESTED PROPERTIES ARE TREATED AS PRIMITIVE VALUES;
    THE STANDARD JSON DECODER IS USED.

    LARGE MANY-PROPERTY OBJECTS CAN BE HANDLED BY `items()`

    :param json:       SOME STRING-LIKE STRUCTURE THAT CAN ASSUME WE LOOK AT
                       ONE CHARACTER AT A TIME, IN ORDER
    :param query_path: A DOT-SEPARATED STRING INDICATING THE PATH TO THE
                       NESTED ARRAY OPTIONALLY, {"items":query_path} TO
                       FURTHER ITERATE OVER PROPERTIES OF OBJECTS FOUND AT
                       query_path
    :param expected_vars: REQUIRED PROPERTY NAMES, USED TO DETERMINE IF
                          MORE-THAN-ONE PASS IS REQUIRED
    :return: RETURNS AN ITERATOR OVER ALL OBJECTS FROM ARRAY LOCATED AT query_path
    """
    if hasattr(json, "read"):
        # ASSUME IT IS A STREAM
        temp = json
        def get_more():
            return temp.read(MIN_READ_SIZE)
        json = List_usingStream(get_more)
    elif hasattr(json, "__call__"):
        json = List_usingStream(json)
    elif isinstance(json, GeneratorType):
        json = List_usingStream(json.next)
    else:
        Log.error("Expecting json to be a stream, or a function that will return more bytes")


    def _iterate_list(index, c, parent_path, path, expected_vars):
        c, index = skip_whitespace(index)
        if c == b']':
            yield index
            return

        while True:
            if not path:
                index = _assign_token(index, c, expected_vars)
                c, index = skip_whitespace(index)
                if c == b']':
                    yield index
                    _done(parent_path)
                    return
                elif c == b',':
                    yield index
                    c, index = skip_whitespace(index)
            else:
                for index in _decode_token(index, c, parent_path, path, expected_vars):
                    c, index = skip_whitespace(index)
                    if c == b']':
                        yield index
                        _done(parent_path)
                        return
                    elif c == b',':
                        yield index
                        c, index = skip_whitespace(index)

    def _done(parent_path):
        if len(parent_path) < len(done[0]):
            done[0] = parent_path

    def _decode_object(index, c, parent_path, query_path, expected_vars):
        if "." in expected_vars:
            if len(done[0]) <= len(parent_path) and all(d == p for d, p in zip(done[0], parent_path)):
                Log.error("Can not pick up more variables, iterator is done")

            if query_path:
                Log.error("Can not extract objects that contain the iteration", var=join_field(query_path))

            index = _assign_token(index, c, expected_vars)
            # c, index = skip_whitespace(index)
            yield index
            return

        did_yield = False
        while True:
            c, index = skip_whitespace(index)
            if c == b',':
                continue
            elif c == b'"':
                name, index = simple_token(index, c)

                c, index = skip_whitespace(index)
                if c != b':':
                    Log.error("Expecting colon")
                c, index = skip_whitespace(index)

                child_expected = needed(name, expected_vars)
                child_path = parent_path + [name]
                if any(child_expected):
                    if not query_path:
                        index = _assign_token(index, c, child_expected)
                    elif query_path[0] == name:
                        for index in _decode_token(index, c, child_path, query_path[1:], child_expected):
                            did_yield = True
                            yield index
                    else:
                        if len(done[0]) <= len(child_path):
                            Log.error("Can not pick up more variables, iterator over {{path}} is done", path=join_field(done[0]))
                        index = _assign_token(index, c, child_expected)
                elif query_path and query_path[0] == name:
                    for index in _decode_token(index, c, child_path, query_path[1:], child_expected):
                        yield index
                else:
                    index = jump_to_end(index, c)
            elif c == b"}":
                if not did_yield:
                    yield index
                break

    def set_destination(expected_vars, value):
        for i, e in enumerate(expected_vars):
            if e is None:
                pass
            elif e == ".":
                destination[i] = value
            elif isinstance(value, Mapping):
                destination[i] = value[e]
            else:
                destination[i] = Null

    def _decode_object_items(index, c, parent_path, query_path, expected_vars):
        """
        ITERATE THROUGH THE PROPERTIES OF AN OBJECT
        """
        c, index = skip_whitespace(index)
        num_items = 0
        while True:
            if c == b',':
                c, index = skip_whitespace(index)
            elif c == b'"':
                name, index = simple_token(index, c)
                if "name" in expected_vars:
                    for i, e in enumerate(expected_vars):
                        if e == "name":
                            destination[i] = name

                c, index = skip_whitespace(index)
                if c != b':':
                    Log.error("Expecting colon")
                c, index = skip_whitespace(index)

                child_expected = needed("value", expected_vars)
                index = _assign_token(index, c, child_expected)
                c, index = skip_whitespace(index)
                if DEBUG and not num_items % 1000:
                    Log.note("{{num}} items iterated", num=num_items)
                yield index
                num_items += 1
            elif c == b"}":
                break

    def _decode_token(index, c, parent_path, query_path, expected_vars):
        if c == b'{':
            if query_path and query_path[0] == "$items":
                if any(expected_vars):
                    for index in _decode_object_items(index, c, parent_path, query_path[1:], expected_vars):
                        yield index
                else:
                    index = jump_to_end(index, c)
                    yield index
            elif not any(expected_vars):
                index = jump_to_end(index, c)
                yield index
            else:
                for index in _decode_object(index, c, parent_path, query_path, expected_vars):
                    yield index
        elif c == b'[':
            for index in _iterate_list(index, c, parent_path, query_path, expected_vars):
                yield index
        else:
            index = _assign_token(index, c, expected_vars)
            yield index

    def _assign_token(index, c, expected_vars):
        if not any(expected_vars):
            return jump_to_end(index, c)

        value, index = simple_token(index, c)
        set_destination(expected_vars, value)

        return index

    def jump_to_end(index, c):
        """
        DO NOT PROCESS THIS JSON OBJECT, JUST RETURN WHERE IT ENDS
        """
        if c == b'"':
            while True:
                c = json[index]
                index += 1
                if c == b'\\':
                    index += 1
                elif c == b'"':
                    break
            return index
        elif c not in b"[{":
            while True:
                c = json[index]
                index += 1
                if c in b',]}':
                    break
            return index - 1

        # OBJECTS AND ARRAYS ARE MORE INVOLVED
        stack = [None] * 1024
        stack[0] = CLOSE[c]
        i = 0  # FOR INDEXING THE STACK
        while True:
            c = json[index]
            index += 1

            if c == b'"':
                while True:
                    c = json[index]
                    index += 1
                    if c == b'\\':
                        index += 1
                    elif c == b'"':
                        break
            elif c in b'[{':
                i += 1
                stack[i] = CLOSE[c]
            elif c == stack[i]:
                i -= 1
                if i == -1:
                    return index  # FOUND THE MATCH!  RETURN
            elif c in b']}':
                Log.error("expecting {{symbol}}", symbol=stack[i])

    def simple_token(index, c):
        if c == b'"':
            json.mark(index - 1)
            while True:
                c = json[index]
                index += 1
                if c == b"\\":
                    index += 1
                elif c == b'"':
                    break
            return json_decoder(json.release(index).decode("utf8")), index
        elif c in b"{[":
            json.mark(index-1)
            index = jump_to_end(index, c)
            value = wrap(json_decoder(json.release(index).decode("utf8")))
            return value, index
        elif c == b"t" and json.slice(index, index + 3) == b"rue":
            return True, index + 3
        elif c == b"n" and json.slice(index, index + 3) == b"ull":
            return None, index + 3
        elif c == b"f" and json.slice(index, index + 4) == b"alse":
            return False, index + 4
        else:
            json.mark(index-1)
            while True:
                c = json[index]
                if c in b',]}':
                    break
                index += 1
            text = json.release(index)
            try:
                return float(text), index
            except Exception:
                Log.error("Not a known JSON primitive: {{text|quote}}", text=text)

    def skip_whitespace(index):
        """
        RETURN NEXT NON-WHITESPACE CHAR, AND ITS INDEX
        """
        c = json[index]
        while c in WHITESPACE:
            index += 1
            c = json[index]
        return c, index + 1

    if isinstance(query_path, Mapping) and query_path.get("items"):
        path_list = split_field(query_path.get("items")) + ["$items"]  # INSERT A MARKER SO THAT OBJECT IS STREAM DECODED
    else:
        path_list = split_field(query_path)

    destination = [None] * len(expected_vars)
    c, index = skip_whitespace(0)
    done = [path_list + [None]]
    for _ in _decode_token(index, c, [], path_list, expected_vars):
        output = Data()
        for i, e in enumerate(expected_vars):
            output[e] = destination[i]
        yield output


def needed(name, required):
    """
    RETURN SUBSET IF name IN REQUIRED
    """
    return [
        relative_field(r, name) if r and startswith_field(r, name) else None
        for r in required
    ]

class List_usingStream(object):
    """
    EXPECTING A FUNCTION
    """
    def __init__(self, get_more_bytes):
        """
        get_more_bytes() SHOULD RETURN AN ARRAY OF BYTES OF ANY SIZE
        """
        if not hasattr(get_more_bytes, "__call__"):
            Log.error("Expecting a function that will return bytes")

        self.get_more = get_more_bytes
        self.start = 0
        self._mark = -1
        self.buffer = self.get_more()
        self.buffer_length = len(self.buffer)
        pass

    def __getitem__(self, index):
        offset = index - self.start
        if offset < len(self.buffer):
            return self.buffer[offset:offset + 1]

        if offset < 0:
            Log.error("Can not go in reverse on stream index=={{index}} (offset={{offset}})", index=index, offset=offset)

        if self._mark == -1:
            self.start += self.buffer_length
            offset = index - self.start
            self.buffer = self.get_more()
            self.buffer_length = len(self.buffer)
            while self.buffer_length <= offset:
                more = self.get_more()
                self.buffer += more
                self.buffer_length = len(self.buffer)
            return self.buffer[offset:offset+1]

        needless_bytes = self._mark - self.start
        if needless_bytes:
            self.start = self._mark
            offset = index - self.start
            self.buffer = self.buffer[needless_bytes:]
            self.buffer_length = len(self.buffer)

        while self.buffer_length <= offset:
            more = self.get_more()
            self.buffer += more
            self.buffer_length = len(self.buffer)

        try:
            return self.buffer[offset:offset+1]
        except Exception as e:
            Log.error("error", cause=e)

    def slice(self, start, stop):
        self.mark(start)
        return self.release(stop)

    def mark(self, index):
        """
        KEEP THIS index IN MEMORY UNTIL release()
        """
        if index < self.start:
            Log.error("Can not go in reverse on stream")
        if self._mark != -1:
            Log.error("Not expected")
        self._mark = index

    def release(self, end):
        if self._mark == -1:
            Log.error("Must mark() this stream before release")

        end_offset = end - self.start
        while self.buffer_length < end_offset:
            self.buffer += self.get_more()
            self.buffer_length = len(self.buffer)

        output = self.buffer[self._mark - self.start:end_offset]
        self._mark = -1
        return output

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

from collections import Mapping

import mo_dots as dot
from mo_dots import Null, Data, FlatList, wrap, wrap_leaves, listwrap
from mo_logs import Log
from mo_math import MAX, OR
from mo_collections.matrix import Matrix
from jx_base.container import Container
from jx_python.cubes.aggs import cube_aggs
from jx_python.lists.aggs import is_aggs
from jx_base.query import _normalize_edge


class Cube(Container):
    """
    A CUBE IS LIKE A NUMPY ARRAY, ONLY WITH THE DIMENSIONS TYPED AND NAMED.
    CUBES ARE BETTER THAN PANDAS BECAUSE THEY DEAL WITH NULLS GRACEFULLY
    """

    def __init__(self, select, edges, data, frum=None):
        """
        data IS EXPECTED TO BE A dict TO MATRICES, BUT OTHER COLLECTIONS ARE
        ALLOWED, USING THE select AND edges TO DESCRIBE THE data
        """

        self.is_value = False if isinstance(select, list) else True
        self.select = select
        self.meta = Data(format="cube")       # PUT EXTRA MARKUP HERE
        self.is_none = False

        if not all(data.values()):
            is_none = True

        # ENSURE frum IS PROPER FORM
        if isinstance(select, list):
            if edges and OR(not isinstance(v, Matrix) for v in data.values()):
                Log.error("Expecting data to be a dict with Matrix values")

        if not edges:
            if not data:
                if isinstance(select, list):
                    Log.error("not expecting a list of records")

                data = {select.name: Matrix.ZERO}
                self.edges = FlatList.EMPTY
            elif isinstance(data, Mapping):
                # EXPECTING NO MORE THAN ONE rownum EDGE IN THE DATA
                length = MAX([len(v) for v in data.values()])
                if length >= 1:
                    self.edges = wrap([{"name": "rownum", "domain": {"type": "rownum"}}])
                else:
                    self.edges = FlatList.EMPTY
            elif isinstance(data, list):
                if isinstance(select, list):
                    Log.error("not expecting a list of records")

                data = {select.name: Matrix.wrap(data)}
                self.edges = wrap([{"name": "rownum", "domain": {"type": "rownum", "min": 0, "max": len(data), "interval": 1}}])
            elif isinstance(data, Matrix):
                if isinstance(select, list):
                    Log.error("not expecting a list of records")

                data = {select.name: data}
            else:
                if isinstance(select, list):
                    Log.error("not expecting a list of records")

                data = {select.name: Matrix(value=data)}
                self.edges = FlatList.EMPTY
        else:
            self.edges = wrap(edges)

        self.data = data

    def __len__(self):
        """
        RETURN DATA VOLUME
        """
        if not self.edges:
            return 1

        return len(self.data.values()[0])

    def __iter__(self):
        if self.is_value:
            return self.data[self.select.name].__iter__()

        if not self.edges:
            return list.__iter__([])

        if len(self.edges) == 1 and wrap(self.edges[0]).domain.type == "index":
            # ITERATE AS LIST OF RECORDS
            keys = list(self.data.keys())
            output = (dot.zip(keys, r) for r in zip(*self.data.values()))
            return output

        Log.error("This is a multicube")

    def query(self, q):
        frum = self
        if is_aggs(q):
            return cube_aggs(frum, q)

        columns = wrap({s.name: s for s in self.select + self.edges})

        # DEFER TO ListContainer
        from jx_python.containers.list_usingPythonList import ListContainer

        frum = ListContainer(name="", data=frum.values(), schema=columns)
        return frum.query(q)

    def values(self):
        """
        TRY NOT TO USE THIS, IT IS SLOW
        """
        matrix = self.data.values()[0]  # CANONICAL REPRESENTATIVE
        if matrix.num == 0:
            return
        e_names = self.edges.name
        s_names = self.select.name
        parts = [e.domain.partitions.value if e.domain.primitive else e.domain.partitions for e in self.edges]
        for c in matrix._all_combos():
            try:
                output = {n: parts[i][c[i]] for i, n in enumerate(e_names)}
            except Exception as e:
                Log.error("problem", cause=e)
            for s in s_names:
                output[s] = self.data[s][c]
            yield wrap(output)

    @property
    def value(self):
        if self.is_none:
            return Null
        if self.edges:
            Log.error("can not get value of with dimension")
        if isinstance(self.select, list):
            Log.error("can not get value of multi-valued cubes")
        return self.data[self.select.name].cube

    def __lt__(self, other):
        return self.value < other

    def __gt__(self, other):
        return self.value > other

    def __eq__(self, other):
        if other == None:
            if self.edges:
                return False
            if self.is_value and self.value == None:
                return True
            return False
        return self.value == other

    def __ne__(self, other):
        return not Cube.__eq__(self, other)

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return other + self.value

    def __sub__(self, other):
        return self.value - other

    def __rsub__(self, other):
        return other - self.value

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    def __div__(self, other):
        return self.value / other

    def __rdiv__(self, other):
        return other / self.value

    def __truediv__(self, other):
        return self.value / other

    def __rtruediv__(self, other):
        return other / self.value

    def __getitem__(self, item):
        # TODO: SOLVE FUNDAMENTAL QUESTION OF IF SELECTING A PART OF AN
        # EDGE REMOVES THAT EDGE FROM THIS RESULT, OR ADDS THE PART
        # AS A select {"name":edge.name, "value":edge.domain.partitions[coord]}
        # PROBABLY NOT, THE value IS IDENTICAL OVER THE REMAINING
        if isinstance(item, Mapping):
            coordinates = [None] * len(self.edges)

            # MAP DICT TO NUMERIC INDICES
            for name, v in item.items():
                ei, parts = wrap([(i, e.domain.partitions) for i, e in enumerate(self.edges) if e.name == name])[0]
                if not parts:
                    Log.error("Can not find {{name}}=={{value|quote}} in list of edges, maybe this feature is not implemented yet",
                        name= name,
                        value= v)
                part = wrap([p for p in parts if p.value == v])[0]
                if not part:
                    return Null
                else:
                    coordinates[ei] = part.dataIndex

            edges = [e for e, v in zip(self.edges, coordinates) if v is None]
            if not edges:
                # ZERO DIMENSIONAL VALUE
                return wrap({k: v.__getitem__(coordinates) for k, v in self.data.items()})
            else:
                output = Cube(
                    select=self.select,
                    edges=wrap([e for e, v in zip(self.edges, coordinates) if v is None]),
                    data={k: Matrix(values=c.__getitem__(coordinates)) for k, c in self.data.items()}
                )
                return output
        elif isinstance(item, text_type):
            # RETURN A VALUE CUBE
            if self.is_value:
                if item != self.select.name:
                    Log.error("{{name}} not found in cube",  name= item)
                return self

            if item not in self.select.name:
                Log.error("{{name}} not found in cube",  name= item)

            output = Cube(
                select=[s for s in self.select if s.name == item][0],
                edges=self.edges,
                data={item: self.data[item]}
            )
            return output
        else:
            Log.error("not implemented yet")

    def __getattr__(self, item):
        return self.data[item]

    def get_columns(self):
        return self.edges + listwrap(self.select)

    def get_leaves(self):
        return self.edges + listwrap(self.select)

    def forall(self, method):
        """
        TODO: I AM NOT HAPPY THAT THIS WILL NOT WORK WELL WITH WINDOW FUNCTIONS
        THE parts GIVE NO INDICATION OF NEXT ITEM OR PREVIOUS ITEM LIKE rownum
        DOES.  MAYBE ALGEBRAIC EDGES SHOULD BE LOOPED DIFFERENTLY?  ON THE
        OTHER HAND, MAYBE WINDOW FUNCTIONS ARE RESPONSIBLE FOR THIS COMPLICATION
        MAR 2015: THE ISSUE IS parts, IT SHOULD BE coord INSTEAD

        IT IS EXPECTED THE method ACCEPTS (value, coord, cube), WHERE
        value - VALUE FOUND AT ELEMENT
        parts - THE ONE PART CORRESPONDING TO EACH EDGE
        cube - THE WHOLE CUBE, FOR USE IN WINDOW FUNCTIONS
        """
        if not self.is_value:
            Log.error("Not dealing with this case yet")

        matrix = self.data.values()[0]
        parts = [e.domain.partitions for e in self.edges]
        for c in matrix._all_combos():
            method(matrix[c], [parts[i][cc] for i, cc in enumerate(c)], self)

    def _select(self, select):
        selects = listwrap(select)
        is_aggregate = OR(s.aggregate != None and s.aggregate != "none" for s in selects)
        if is_aggregate:
            values = {s.name: Matrix(value=self.data[s.value].aggregate(s.aggregate)) for s in selects}
            return Cube(select, [], values)
        else:
            values = {s.name: self.data[s.value] for s in selects}
            return Cube(select, self.edges, values)

    def filter(self, where):
        if len(self.edges)==1 and self.edges[0].domain.type=="index":
            # USE THE STANDARD LIST FILTER
            from jx_python import jx
            return jx.filter(self.data.values()[0].cube, where)
        else:
            # FILTER DOES NOT ALTER DIMESIONS, JUST WHETHER THERE ARE VALUES IN THE CELLS
            Log.unexpected("Incomplete")

    def _groupby(self, edges):
        """
        RETURNS LIST OF (coord, values) TUPLES, WHERE
            coord IS THE INDEX INTO self CUBE (-1 INDEX FOR COORDINATES NOT GROUPED BY)
            values ALL VALUES THAT BELONG TO THE SLICE

        """
        edges = FlatList([n for e in edges for n in _normalize_edge(e)])

        stacked = [e for e in self.edges if e.name in edges.name]
        remainder = [e for e in self.edges if e.name not in edges.name]
        selector = [1 if e.name in edges.name else 0 for e in self.edges]

        if len(stacked) + len(remainder) != len(self.edges):
            Log.error("can not find some edges to group by")
        # CACHE SOME RESULTS
        keys = edges.name
        getKey = [e.domain.getKey for e in self.edges]
        lookup = [[getKey[i](p) for p in e.domain.partitions+([None] if e.allowNulls else [])] for i, e in enumerate(self.edges)]

        if isinstance(self.select, list):
            selects = listwrap(self.select)
            index, v = zip(*self.data[selects[0].name].groupby(selector))

            coord = wrap([coord2term(c) for c in index])

            values = [v]
            for s in selects[1::]:
                i, v = zip(*self.data[s.name].group_by(selector))
                values.append(v)

            output = zip(coord, [Cube(self.select, remainder, {s.name: v[i] for i, s in enumerate(selects)}) for v in zip(*values)])
        elif not remainder:
            # v IS A VALUE, NO NEED TO WRAP IT IN A Cube
            output = (
                (
                    coord2term(coord),
                    v
                )
                for coord, v in self.data[self.select.name].groupby(selector)
            )
        else:
            output = (
                (
                    coord2term(coord),
                    Cube(self.select, remainder, v)
                )
                for coord, v in self.data[self.select.name].groupby(selector)
            )

        return output


    def groupby(self, edges):
        """
        SLICE THIS CUBE IN TO ONES WITH LESS DIMENSIONALITY
        simple==True WILL HAVE GROUPS BASED ON PARTITION VALUE, NOT PARTITION OBJECTS
        """
        edges = FlatList([n for e in edges for n in _normalize_edge(e) ])

        stacked = [e for e in self.edges if e.name in edges.name]
        remainder = [e for e in self.edges if e.name not in edges.name]
        selector = [1 if e.name in edges.name else 0 for e in self.edges]

        if len(stacked) + len(remainder) != len(self.edges):
            Log.error("can not find some edges to group by")
        # CACHE SOME RESULTS
        keys = edges.name
        getKey = [e.domain.getKey for e in self.edges]
        lookup = [[getKey[i](p) for p in e.domain.partitions+([None] if e.allowNulls else [])] for i, e in enumerate(self.edges)]

        def coord2term(coord):
            output = wrap_leaves({keys[i]: lookup[i][c] for i, c in enumerate(coord)})
            return output

        if isinstance(self.select, list):
            selects = listwrap(self.select)
            index, v = zip(*self.data[selects[0].name].groupby(selector))

            coord = wrap([coord2term(c) for c in index])

            values = [v]
            for s in selects[1::]:
                i, v = zip(*self.data[s.name].group_by(selector))
                values.append(v)

            output = zip(coord, [Cube(self.select, remainder, {s.name: v[i] for i, s in enumerate(selects)}) for v in zip(*values)])
        elif not remainder:
            # v IS A VALUE, NO NEED TO WRAP IT IN A Cube
            output = (
                (
                    coord2term(coord),
                    v
                )
                for coord, v in self.data[self.select.name].groupby(selector)
            )
        else:
            output = (
                (
                    coord2term(coord),
                    Cube(self.select, remainder, v)
                )
                for coord, v in self.data[self.select.name].groupby(selector)
            )

        return output

    def window(self, window):
        if window.edges or window.sort:
            Log.error("not implemented")

        from jx_python import jx

        # SET OP
        canonical = self.data.values()[0]
        accessor = jx.get(window.value)
        cnames = self.data.keys()

        # ANNOTATE EXISTING CUBE WITH NEW COLUMN
        m = self.data[window.name] = Matrix(dims=canonical.dims)
        for coord in canonical._all_combos():
            row = Data()  # IT IS SAD WE MUST HAVE A Data(), THERE ARE {"script": expression} USING THE DOT NOTATION
            for k in cnames:
                row[k] = self.data[k][coord]
            for c, e in zip(coord, self.edges):
                row[e.name] = e.domain.partitions[c]
            m[coord] = accessor(row, Null, Null)  # DUMMY Null VALUES BECAUSE I DO NOT KNOW WHAT TO DO

        self.select.append(window)
        return self

    def format(self, format):
        if format == None or format == "cube":
            return self
        else:
            Log.error("Do not know how to handle")

    def __str__(self):
        if self.is_value:
            return str(self.data)
        else:
            return str(self.data)

    def __int__(self):
        if self.is_value:
            return int(self.value)
        else:
            return int(self.data)

    def __float__(self):
        if self.is_value:
            v = self.value
            if v == None:
                return v
            else:
                return float(v)
        else:
            return float(self.data)

    def __data__(self):
        return Data(
            select=self.select,
            edges=self.edges,
            data={k: v.cube for k, v in self.data.items()},
            meta=self.meta
        )

"""Utils for working with the parquet thrift models."""
from collections import OrderedDict

from fastparquet import parquet_thrift


def schema_tree(schema, i=0, paths={}, path=[]):
    root = schema[i]
    if i :
        path = path + [root.name]
        paths[".".join(path)] = root
    root["children"] = OrderedDict()
    while len(root["children"]) < root.num_children:
        i += 1
        s = schema[i]
        s["parent"] = root
        root["children"][s.name] = s
        if s.num_children not in [None, 0]:
            i = schema_tree(schema, i, paths, path)
        else:
            paths[".".join(path + [s.name])] = s
    if root.num_children:
        return i
    else:
        return i + 1


def schema_to_text(root, indent=[]):
    text = "".join(indent) + '- ' + root.name + ": "
    parts = []
    if root.type is not None:
        parts.append(parquet_thrift.Type._VALUES_TO_NAMES[root.type])
    if root.logicalType is not None:
        for key in dir(root.logicalType):
            if getattr(root.logicalType, key) is not None:
                if key == "TIMESTAMP":
                    unit = [k for k in dir(root.logicalType.TIMESTAMP.unit) if getattr(
                        root.logicalType.TIMESTAMP.unit, k) is not None][0]
                    parts.append(f"TIMESTAMP[{unit}]")
                else:
                    # extra parameters possible here
                    parts.append(key)
                break

    if root.converted_type is not None:
        parts.append(parquet_thrift.ConvertedType._VALUES_TO_NAMES[
                         root.converted_type])
    if root.repetition_type is not None:
        parts.append(parquet_thrift.FieldRepetitionType._VALUES_TO_NAMES[
                         root.repetition_type])
    text += ', '.join(parts)
    indent.append('|')
    if hasattr(root, 'children'):
        indent[-1] = '| '
        for i, child in enumerate(root["children"].values()):
            if i == len(root["children"]) - 1:
                indent[-1] = '  '
            text += '\n' + schema_to_text(child, indent)
    indent.pop()
    return text


class SchemaHelper(object):
    """Utility providing convenience methods for schema_elements."""

    def __init__(self, schema_elements):
        """Initialize with the specified schema_elements."""
        self.schema_elements = schema_elements
        for se in schema_elements:
            try:
                se[4] = se[4].decode()
            except AttributeError:
                pass  # already a str
        self.root = schema_elements[0]
        self.schema_elements_by_name = dict(
            [(se[4], se) for se in schema_elements])
        self.tree = {}
        schema_tree(schema_elements, paths = self.tree)
        self._text = None

    @property
    def text(self):
        if self._text is None:
            self._text = schema_to_text(self.schema_elements[0])
        return self._text

    def __eq__(self, other):
        return self.schema_elements == other.schema_elements

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.text

    def __repr__(self):
        return f"<Parquet Schema with {self.schema_elements} entries>"

    def schema_element(self, name):
        """Get the schema element with the given name or path"""
        if not isinstance(name, str):
            name = ".".join(name)
        return self.tree[name]

    def __getitem__(self, item):
        return self.schema_element(item)

    def is_required(self, name):
        """Return true if the schema element with the given name is required."""
        required = True
        if isinstance(name, str):
            name = name.split('.')
        parts = []
        for part in name:
            parts.append(part)
            s = self.schema_element(parts)
            if s.repetition_type != parquet_thrift.FieldRepetitionType.REQUIRED:
                required = False
                break
        return required

    def max_repetition_level(self, parts):
        """Get the max repetition level for the given schema path."""
        max_level = 0
        if isinstance(parts, str):
            parts = parts.split('.')
        for i in range(len(parts)):
            element = self.schema_element(parts[:i+1])
            if element.repetition_type == parquet_thrift.FieldRepetitionType.REPEATED:
                max_level += 1
        return max_level

    def max_definition_level(self, parts):
        """Get the max definition level for the given schema path."""
        max_level = 0
        if isinstance(parts, str):
            parts = parts.split('.')
        for i in range(len(parts)):
            element = self.schema_element(parts[:i+1])
            if element.repetition_type != parquet_thrift.FieldRepetitionType.REQUIRED:
                max_level += 1
        return max_level

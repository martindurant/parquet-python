from copy import copy
import json
import struct

from .thrift_structures import write_thrift
from .util import default_open

MARKER = b'PAR1'


def consolidate_categories(fmd):
    key_value = [k for k in fmd.key_value_metadata
                 if k.key == 'pandas'][0]
    meta = json.loads(key_value.value)
    cats = [c for c in meta['columns']
            if 'num_categories' in (c['metadata'] or [])]
    for cat in cats:
        for rg in fmd.row_groups:
            for col in rg.columns:
                if ".".join(col.meta_data.path_in_schema) == cat['name']:
                    ncats = [k.value for k in (col.meta_data.key_value_metadata or [])
                             if k.key == 'num_categories']
                    if ncats and int(ncats[0]) > cat['metadata'][
                            'num_categories']:
                        cat['metadata']['num_categories'] = int(ncats[0])
    key_value.value = json.dumps(meta, sort_keys=True)

def write_common_metadata(fn, fmd, open_with=default_open,
                          no_row_groups=True):
    """
    For hive-style parquet, write schema in special shared file

    Parameters
    ----------
    fn: str
        Filename to write to
    fmd: thrift FileMetaData
        Information to write
    open_with: func
        To use to create writable file as f(path, mode)
    no_row_groups: bool (True)
        Strip out row groups from metadata before writing - used for "common
        metadata" files, containing only the schema.
    """
    consolidate_categories(fmd)
    with open_with(fn, 'wb') as f:
        f.write(MARKER)
        if no_row_groups:
            fmd = copy(fmd)
            fmd.row_groups = []
        foot_size = write_thrift(f, fmd)
        f.write(struct.pack(b"<i", foot_size))
        f.write(MARKER)

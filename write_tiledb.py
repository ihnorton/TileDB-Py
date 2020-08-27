#%%
import tiledb
from tiledb import Attr, FilterList, ArraySchema, Domain, Dim, ZstdFilter
import pyarrow, pyarrow.csv, pyarrow.parquet
import pyarrow as pa
import numpy as np, pandas as pd
import os, tempfile, time, glob
import uuid
import io
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pdb

print("loaded! ", os.getpid())

if __name__ == '__main__':
    # important!
    if multiprocessing.get_start_method(True) != 'spawn':
        multiprocessing.set_start_method('spawn')


def create_schema():
    schema = ArraySchema(
          domain=Domain(*[
            Dim(name='longitude', domain=(-1.7976931348623157e+308, 1.7976931348623157e+308), tile=10000, dtype='float64'),
            Dim(name='latitude', domain=(-1.7976931348623157e+308, 1.7976931348623157e+308), tile=10000, dtype='float64'),
          ]),
          attrs=[
            Attr(name='FID', dtype='<U0', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='mmsi', dtype='int64', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='imo', dtype='int64', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='vessel_name', dtype='<U0', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='callsign', dtype='<U0', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='vessel_type', dtype='<U0', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='vessel_type_code', dtype='<U0', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='vessel_type_cargo', dtype='<U0', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='vessel_class', dtype='<U0', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='length', dtype='int64', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='width', dtype='int64', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='flag_country', dtype='<U0', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='flag_code', dtype='<U0', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='destination', dtype='<U0', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='eta', dtype='<U0', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='draught', dtype='float64', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='sog', dtype='float64', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='cog', dtype='float64', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='rot', dtype='float64', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='heading', dtype='int64', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='nav_status', dtype='<U0', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='nav_status_code', dtype='<U0', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='source', dtype='<U0', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='ts_pos_utc', dtype='datetime64[ns]', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='ts_static_utc', dtype='datetime64[ns]', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='dt_pos_utc', dtype='datetime64[ns]', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='dt_static_utc', dtype='datetime64[ns]', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='vessel_type_main', dtype='<U0', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='vessel_type_sub', dtype='<U0', filters=FilterList([ZstdFilter(level=7), ])),
            Attr(name='message_type', dtype='int64', filters=FilterList([ZstdFilter(level=7), ])),
          ],
          cell_order='row-major',
          tile_order='row-major',
          capacity=10000,
          sparse=True,
          allows_duplicates=True,
          coords_filters=FilterList([ZstdFilter(level=7), ])
        )
    return schema

def create_array(uri):
    schema = create_schema()

    tiledb.Array.create(uri, schema)

#%%
def convert_pqt_partition(path, count=-1):
    #import pdb; pdb.set_trace()
    input_files = glob.glob(path + "/*.pqt")

    if count == -1:
        count = len(input_files)

    if len(input_files) < 1:
        raise ValueError(f"no input files for path '{path}'!")

    tbs = []
    for pqt_file in input_files[:count]:
        tbs.append(pa.parquet.read_pandas(pqt_file))

    #import pdb; pdb.set_trace()
    table = pa.concat_tables(tbs)
    df = table.to_pandas()

    return df

def write_df_tiledb(uri, df, source=None, cfg=None):
    cfg = cfg if cfg is not None else {}
    ctx = tiledb.Ctx(cfg)
    #pdb.set_trace()

    cols = list(df.columns)
    cols.remove('longitude')
    cols.remove('latitude')

    data = {
        name: np.array(df[name]) for name in cols
    }

    lon_coords = np.array(df.longitude)
    lat_coords = np.array(df.latitude)

    print(f"writing '{uri}'")
    with tiledb.open(uri, 'w', ctx=ctx) as A:
        A[lon_coords, lat_coords] = data

        if source is not None:
            A.meta['partition_source'] = source

def do_convert(output_array, subset_path, subset_count=-1, cfg=None):
    if not os.path.isdir(output_array):
        raise ValueError(f"output_array path does not exist {output_array}")

    # test code
    #tmpp = os.path.join(output_array, str(uuid.uuid1()))
    #import io
    #with io.open(tmpp, 'w') as f:
    #    f.write(''.join([path, ' ', output_array, ' ', str(subset_count)]))

    try:
        df = convert_pqt_partition(subset_path, count=subset_count)
        write_df_tiledb(output_array, df, source=str(subset_path))

        subset_name = os.path.basename(os.path.abspath(subset_path))
        done_file = os.path.join(output_array, subset_name + ".done")
        if os.path.isfile(done_file):
            raise FileExistsError(f".done file should not exist for this path '{done_file}'")

        with io.open(done_file, 'w') as f:
            f.write("")

    except Exception as e:
        return (e, subset_path)

    return None

#%%
tasks = []

# this must be a separate cell because python import system is broken
def run_mp(output_array):
    pool_dir = "/test_deleteme19Aug2020/test/tmp_pool_hold"
    if not os.path.isdir(pool_dir):
        raise Exception("pool_dir does not exist")

    ############################################################

    append = False

    nproc = 2
    #subset_count = 1
    subset_count = -1

    # TODO these should have a suffix for filtering
    input_partitions = glob.glob(pool_dir + "/*_*/")

    cfg = {}

    ############################################################

    if not append:
        if os.path.isdir(output_array):
            raise Exception("output_array already exists")

        create_array(output_array)

    with ProcessPoolExecutor(max_workers=nproc) as executor:
        # TODO save the fragment id somewhere

        for pt_idx in range(0,len(input_partitions)):
            subset_path = input_partitions[pt_idx]

            print(f"running: {subset_path}")

            task = executor.submit(
                do_convert,
                *(output_array, subset_path),
                **dict(subset_count=subset_count),
            )
            tasks.append(task)

    for t in tasks:
        result = t.result()
        if result == None:
            continue
        elif isinstance(result[0], Exception):
            print("Error: ", result)
        else:
            print(result)


# %%
output_array = "/test_deleteme19Aug2020/test/full_array.tdb"
#output_array = "/test_deleteme19Aug2020/test/subset_1.tdb"

if __name__ == '__main__':
## #! %time
    start = time.time()
    run_mp(output_array)
    duration = time.time()-start
    print("duration: ", duration)
# %%

# %%

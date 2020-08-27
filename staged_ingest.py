#%%

import tiledb
import pyarrow, pyarrow.csv, pyarrow.parquet
import pyarrow as pa
import numpy as np, pandas as pd
import os, tempfile, time, glob
import uuid
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

#%%
if __name__ == '__main__':
    # important!
    multiprocessing.set_start_method('spawn')


#%%
metadata = {b'pandas': b'{"index_columns": [{"kind": "range", "name": null, "start": 0, "stop": 297569, "step": 1}], "column_indexes": [{"name": null, "field_name": null, "pandas_type": "unicode", "numpy_type": "object", "metadata": {"encoding": "UTF-8"}}], "columns": [{"name": "FID", "field_name": "FID", "pandas_type": "unicode", "numpy_type": "object", "metadata": null}, {"name": "mmsi", "field_name": "mmsi", "pandas_type": "int64", "numpy_type": "int64", "metadata": null}, {"name": "imo", "field_name": "imo", "pandas_type": "int64", "numpy_type": "int64", "metadata": null}, {"name": "vessel_name", "field_name": "vessel_name", "pandas_type": "unicode", "numpy_type": "object", "metadata": null}, {"name": "callsign", "field_name": "callsign", "pandas_type": "unicode", "numpy_type": "object", "metadata": null}, {"name": "vessel_type", "field_name": "vessel_type", "pandas_type": "unicode", "numpy_type": "object", "metadata": null}, {"name": "vessel_type_code", "field_name": "vessel_type_code", "pandas_type": "unicode", "numpy_type": "object", "metadata": null}, {"name": "vessel_type_cargo", "field_name": "vessel_type_cargo", "pandas_type": "unicode", "numpy_type": "object", "metadata": null}, {"name": "vessel_class", "field_name": "vessel_class", "pandas_type": "unicode", "numpy_type": "object", "metadata": null}, {"name": "length", "field_name": "length", "pandas_type": "int64", "numpy_type": "int64", "metadata": null}, {"name": "width", "field_name": "width", "pandas_type": "int64", "numpy_type": "int64", "metadata": null}, {"name": "flag_country", "field_name": "flag_country", "pandas_type": "unicode", "numpy_type": "object", "metadata": null}, {"name": "flag_code", "field_name": "flag_code", "pandas_type": "unicode", "numpy_type": "object", "metadata": null}, {"name": "destination", "field_name": "destination", "pandas_type": "unicode", "numpy_type": "object", "metadata": null}, {"name": "eta", "field_name": "eta", "pandas_type": "unicode", "numpy_type": "object", "metadata": null}, {"name": "draught", "field_name": "draught", "pandas_type": "float64", "numpy_type": "float64", "metadata": null}, {"name": "sog", "field_name": "sog", "pandas_type": "float64", "numpy_type": "float64", "metadata": null}, {"name": "cog", "field_name": "cog", "pandas_type": "float64", "numpy_type": "float64", "metadata": null}, {"name": "rot", "field_name": "rot", "pandas_type": "float64", "numpy_type": "float64", "metadata": null}, {"name": "heading", "field_name": "heading", "pandas_type": "int64", "numpy_type": "int64", "metadata": null}, {"name": "nav_status", "field_name": "nav_status", "pandas_type": "unicode", "numpy_type": "object", "metadata": null}, {"name": "nav_status_code", "field_name": "nav_status_code", "pandas_type": "unicode", "numpy_type": "object", "metadata": null}, {"name": "source", "field_name": "source", "pandas_type": "unicode", "numpy_type": "object", "metadata": null}, {"name": "ts_pos_utc", "field_name": "ts_pos_utc", "pandas_type": "datetime", "numpy_type": "datetime64[ns]", "metadata": null}, {"name": "ts_static_utc", "field_name": "ts_static_utc", "pandas_type": "datetime", "numpy_type": "datetime64[ns]", "metadata": null}, {"name": "dt_pos_utc", "field_name": "dt_pos_utc", "pandas_type": "datetime", "numpy_type": "datetime64[ns]", "metadata": null}, {"name": "dt_static_utc", "field_name": "dt_static_utc", "pandas_type": "datetime", "numpy_type": "datetime64[ns]", "metadata": null}, {"name": "vessel_type_main", "field_name": "vessel_type_main", "pandas_type": "unicode", "numpy_type": "object", "metadata": null}, {"name": "vessel_type_sub", "field_name": "vessel_type_sub", "pandas_type": "unicode", "numpy_type": "object", "metadata": null}, {"name": "message_type", "field_name": "message_type", "pandas_type": "int64", "numpy_type": "int64", "metadata": null}, {"name": "longitude", "field_name": "longitude", "pandas_type": "float64", "numpy_type": "float64", "metadata": null}, {"name": "latitude", "field_name": "latitude", "pandas_type": "float64", "numpy_type": "float64", "metadata": null}], "creator": {"library": "pyarrow", "version": "1.0.0"}, "pandas_version": "1.0.3"}'}
fields = [
    pyarrow.field('FID', 'string', False),
    pyarrow.field('mmsi', 'int64', False),
    pyarrow.field('imo', 'int64', False),
    pyarrow.field('vessel_name', 'string', False),
    pyarrow.field('callsign', 'string', False),
    pyarrow.field('vessel_type', 'string', False),
    pyarrow.field('vessel_type_code', 'string', False),
    pyarrow.field('vessel_type_cargo', 'string', False),
    pyarrow.field('vessel_class', 'string', False),
    pyarrow.field('length', 'int64', False),
    pyarrow.field('width', 'int64', False),
    pyarrow.field('flag_country', 'string', False),
    pyarrow.field('flag_code', 'string', False),
    pyarrow.field('destination', 'string', False),
    pyarrow.field('eta', 'string', False),
    pyarrow.field('draught', 'double', False),
    pyarrow.field('sog', 'double', False),
    pyarrow.field('cog', 'double', False),
    pyarrow.field('rot', 'double', False),
    pyarrow.field('heading', 'int64', False),
    pyarrow.field('nav_status', 'string', False),
    pyarrow.field('nav_status_code', 'string', False),
    pyarrow.field('source', 'string', False),
    pyarrow.field('ts_pos_utc', 'timestamp[s]', False),
    pyarrow.field('ts_static_utc', 'timestamp[s]', False),
    pyarrow.field('dt_pos_utc', 'timestamp[s]', False),
    pyarrow.field('dt_static_utc', 'timestamp[s]', False),
    pyarrow.field('vessel_type_main', 'string', False),
    pyarrow.field('vessel_type_sub', 'string', False),
    pyarrow.field('message_type', 'int64', False),
    pyarrow.field('longitude', 'double', False),
    pyarrow.field('latitude', 'double', False)
]
schema = pa.schema(fields, metadata=metadata)

# %%
# dt column format examples
# ts_: 20190802193001
# ds_:  2019-08-02 19:30:0
ts_parsers = ["%Y%m%d%H%M%S", "%Y-%m-%d %H:%M:%S"]
csv_conv_opts = pa.csv.ConvertOptions(column_types=schema, timestamp_parsers=ts_parsers)

#%%
#t = pa.csv.read_csv#("../subset_100/exactEarth_20190802193000_20190802200000.csv",
#                    convert_options=csv_conv_opts)
#def run_all(input_csvs):
#    pool_dir = "/test_deleteme19Aug2020/test/tmp_pool"
#    if os.path.isdir(pool_dir):
#        raise Exception("pool_dir exists")
#    step = 5
#    for csv_idx in range(0, len(input_csvs), 10):
#        run_csvs(input_csvs[csv_idx:csv_idx+step])

# %%
def filter_df_lon(df, lon0, lon1):
    ln = df.longitude
    if lon0 == -180:
        idx = np.where((ln >= lon0) & (ln <= lon1))
    else:
        idx = np.where((ln > lon0) & (ln <= lon1))

    return df.loc[idx]

def run_csvs(csvs, subset_uuid):
    print("running: ", csvs)
    #import pdb; pdb.set_trace()
    pool_dir = "/test_deleteme19Aug2020/test/tmp_pool"
    if not os.path.isdir(pool_dir):
        os.mkdir(pool_dir)

    input_tables = []
    for csv in csvs:
        input_t = pa.csv.read_csv(csv, convert_options=csv_conv_opts)
        input_tables.append(input_t)

    t = pa.concat_tables(input_tables)
    df = t.to_pandas()

    basepath = str(subset_uuid)

    for i in range(-180, 180, 5):
        ln0, ln1 = i, i+5
        subset = filter_df_lon(df,ln0, ln1)

        subset_dir = os.path.join(pool_dir, f"{ln0}_{ln1}")
        if not os.path.isdir(subset_dir):
            os.mkdir(subset_dir)
        subset_path = os.path.join(subset_dir, basepath+".pqt")

        subset_table = pa.Table.from_pandas(subset)

        pa.parquet.write_table(subset_table, subset_path)
        print('.', sep='', end='')

    with open(os.path.join(pool_dir, basepath+".csv_list"), "w") as f:
        f.writelines([c+"\n" for c in csvs])
    print("---")

# %%
def run_mp():
    pool_dir = "/test_deleteme19Aug2020/test/tmp_pool"
    if os.path.isdir(pool_dir):
        raise Exception("pool_dir exists")

    #input_csvs = glob.glob("../subset_100/*.csv")
    input_csvs = glob.glob("/test_deleteme19Aug2020/exactearth_201908/*.csv")

    #import pdb; pdb.set_trace()
    nproc = 24
    step = 10
    with ProcessPoolExecutor(max_workers=nproc) as executor:
        for csv_idx in range(0,len(input_csvs), step):
            last = csv_idx + step
            if last > len(input_csvs):
                last = len(input_csvs)
            subset = input_csvs[csv_idx:last]
            # generate a unique UUID for this subset for tracing
            subset_uuid = uuid.uuid1()

            print(f"running: {subset_uuid}, {len(subset)} {subset[0]} .. {subset[-1]}")

            task = executor.submit(
                run_csvs,
                *(subset, subset_uuid),
            )
# %%
# %%

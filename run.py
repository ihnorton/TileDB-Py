#if __name__ == '__main__':
#%%
import sys
import numpy as np
import pandas as pd
import tiledb

sys.path.append("../TileDB-Py/examples")
from parallel_csv_ingestion import from_csv_mp


# %%
attr_types = {
  "FID":                            np.str,
  "mmsi":                           pd.Int64Dtype(),
  "imo":                            pd.Int64Dtype(),
  "vessel_name":                    np.str,
  "callsign":                       np.str,
  "vessel_type":                    np.str,
  "vessel_type_code":               np.str,
  "vessel_type_cargo":              np.str,
  "vessel_class":                   np.str,
  "length":                         pd.Int64Dtype(),
  "width":                          pd.Int64Dtype(),
  "flag_country":                   np.str,
  "flag_code":                      np.str,
  "destination":                    np.str,
#  "eta":                            np.int64,
  "draught":                        np.float64,
  "sog":                            np.float64,
  "cog":                            np.float64,
  "rot":                            np.float64,
  "heading":                        pd.Int64Dtype(),
  "nav_status":                     np.str,
  "nav_status_code":                np.str,
  "source":                         np.str,
#  "ts_pos_utc":                     'datetime64[ns]',
#  "ts_static_utc":                  'datetime64[ns]',
#  "dt_pos_utc":                     'datetime64[ns]',
#  "dt_static_utc":                  'datetime64[ns]',
  "vessel_type_main":               np.str,
  "vessel_type_sub":                np.str,
  "message_type":                   pd.Int64Dtype()
}

fillna = dict()
for name, dtype in attr_types.items():
    if dtype == np.str:
        fillna[name] = ''
    elif dtype == pd.Int64Dtype():
        fillna[name] = 0

csv_path = "/test_vol_10Oct2020/exactearth_201908/"
array_path = "/test_vol_10Oct2020/data/subset_200_nproc24_chunk100k_step4/"

chunksize = 100_000
csv_list_step = 4
nproc = 24

if __name__ == '__main__':
  from_csv_mp(csv_path,
              array_path,
              chunksize=chunksize,
              list_step_size=csv_list_step,
              max_workers=nproc,
              sparse=True,
              index_col = ['longitude', 'latitude'],
              parse_dates=['ts_pos_utc', 'ts_static_utc', 'dt_pos_utc', 'dt_static_utc', 'eta'],
              attr_types=attr_types,
              fillna=fillna,
              debug=True)
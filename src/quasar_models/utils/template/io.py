__all__ = ['drop_nonpos', 'drop_nonfinite', 'get_table_data']

from numpy import isfinite
from astropy.io.fits import BinTableHDU
from astropy.units import Quantity

from quasar_typing.numpy import FloatVector

def drop_nonpos(arr: FloatVector) -> FloatVector:
    return arr[arr > 0]

def drop_nonfinite(arr: FloatVector) -> FloatVector:
    return arr[isfinite(arr)]

def get_table_data(table: BinTableHDU, cname: str) -> FloatVector | Quantity:
    data = table.data[cname]
    col_index = table.columns.names.index(cname)
    col = table.columns[col_index]
    n = col.array.shape[0]
    return data[:n]
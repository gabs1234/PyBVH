import numpy as np

float4_dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32), ('w', np.float32)])

def make_float4(x, y, z, w) -> np.ndarray:
    try:
        return np.array([(x, y, z, w)], dtype=float4_dtype)
    except ValueError:
        pass
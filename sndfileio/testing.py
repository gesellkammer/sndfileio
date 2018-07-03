from typing import Tuple
import numpy as np
from .sndfileio import BACKENDS


def compare_read(path:str, backend) -> Tuple[np.ndarray, np.ndarray]:
    backends = [backend for backend in BACKENDS if backend.is_available()]
    backend1 = next(b for b in backends if b.name == backend)
    backend2 = next(b for b in backends if b.name == 'builtin')
    s0 = backend1.read(path)
    s1 = backend2.read(path)
    return s0, s1


def testfile(path:str, exact=True) -> bool:
    s0, s1 = compare_read(path)
    if exact:
        return (s0[0] == s1[0]).all()
    return np.allclose(s0, s1)
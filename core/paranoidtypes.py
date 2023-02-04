import paranoid as pns
import numpy as np

class SymmetricMatrix(pns.Type):
    def test(self, m):
        assert isinstance(m, np.ndarray), "Not an ndarray"
        assert len(m.shape) == 2, "Not of dimension 2"
        assert m.shape[0] == m.shape[1], "Not square matrix"
        assert np.all(m == m.T), "Not a perfectly symmetric matrix"
        assert np.all(np.isfinite(m)), "Must not contain NaN or inf"
    def generate(self):
        pass

class CorrelationMatrix(SymmetricMatrix):
    """A pseudo-correlation matrix.

    Must contain approximately 1's on the diagonal, contain values
    between -1 and 1, and be exactly equal to its transpose.
    """
    def test(self, m):
        super().test(m)
        assert np.allclose(np.diag(m), 1), "Diagonal contains numbers other than 1."
        assert np.max(m) <= 1, "Contains value greater than 1"
        assert np.min(m) >= -1, "Contains value less than -1"

class CovarianceMatrix(SymmetricMatrix):
    def test(self, m):
        super().test(m)
        assert np.min(np.linalg.eig(m)[0]) > -1e-10, "Not all eigenvalues are non-negative"

class TrueCorrelationMatrix(CorrelationMatrix):
    def test(self, m):
        super().test(m)
        assert np.min(np.linalg.eig(m)[0]) > -1e-10, "Not all eigenvalues are non-negative"

class DistanceMatrix(SymmetricMatrix):
    def test(self, m):
        super().test(m)
        assert np.min(m) >= 0, "Negative distance is not allowed"
        assert np.allclose(np.diag(m), 0), "Diagonal contains numbers other than 0."

class ParamsList(pns.Type):
    def __init__(self, params=None):
        self.params = params
        if params is not None:
            super().__init__(params)
        else:
            super().__init__()
    def test(self, v):
        assert v in pns.Dict(pns.String, pns.Number), "Invalid param format"
        if self.params is not None:
            assert set(v.keys()) == set(self.params.keys()), "Not all params specified"
            assert all(self.params[p][0] <= v[p] and v[p] <= self.params[p][1] for p in v.keys())

class Graph(pns.Type):
    def test(self, v):
        assert v in pns.NDArray(d=2, t=pns.Set([0, 1])), "Invalid adjacency matrix"
        assert np.all(v == v.T), "Directed graph"

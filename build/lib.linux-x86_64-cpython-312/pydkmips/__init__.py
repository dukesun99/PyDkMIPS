import numpy as np
from ._pydkmips_impl import Greedy as _Greedy, Dual_Greedy as _Dual_Greedy, BC_Greedy as _BC_Greedy, BC_Dual as _BC_Dual

class DkMIPS:
    """
    Abstract class for DkMIPS algorithms.
    """
    def __init__(self, d1, d2=None, impl_class=None, is_bc_method=False):
        if d2 is None:
            self.d = d1
            self._is_two_space = False
        else:
            self.d1 = d1
            self.d2 = d2
            self._is_two_space = True
        self._impl = None
        self.data = None
        self.data_2 = None
        self._impl_class = impl_class
        self._is_bc_method = is_bc_method
    def _add_single(self, items):
        """
        Add items to the DkMIPS instance.
        
        Args:
            items (numpy.ndarray): Array of items with shape (n, d) and dtype float32
        
        Raises:
            ValueError: If input parameters are invalid
        """
        if self._is_two_space:
            raise ValueError("Please use the overloaded add function for two-space case")
        
        # Input validation
        if not isinstance(items, np.ndarray):
            raise ValueError("items must be a numpy array")
        
        if items.dtype != np.float32:
            items = items.astype(np.float32)
        
        if len(items.shape) != 2:
            raise ValueError(f"items must be 2D array, got shape {items.shape}")
        
        if items.shape[1] != self.d:
            raise ValueError(f"items dimension must be ({self.d}), got {items.shape[1]}")
        
        if self._impl is not None or self.data is not None:
            raise ValueError("Greedy instance already initialized. We don't support adding items after initialization at the moment.")
        
        self.n = items.shape[0]
        
        self._impl = self._impl_class(self.n, self.d, self.d, items, items)
        
        self.data = items
    
    def _add_dual(self, items1, items2):
        if not self._is_two_space:
            raise ValueError("Please use the overloaded add function for single-space case")
        
        if not isinstance(items1, np.ndarray) or not isinstance(items2, np.ndarray):
            raise ValueError("items1 and items2 must be numpy arrays")
        
        if items1.dtype != np.float32 or items2.dtype != np.float32:
            items1 = items1.astype(np.float32)
            items2 = items2.astype(np.float32)
        
        if len(items1.shape) != 2 or len(items2.shape) != 2:
            raise ValueError("items1 and items2 must be 2D arrays")
        
        if items1.shape[1] != self.d1 or items2.shape[1] != self.d2:
            raise ValueError(f"items1 dimension must be ({self.d1}), items2 dimension must be ({self.d2})")
        
        if self._impl is not None or self.data is not None:
            raise ValueError("Greedy instance already initialized. We don't support adding items after initialization at the moment.")
        
        self.n = items1.shape[0]
        
        if items1.shape[0] != items2.shape[0] or items2.shape[0] != self.n:
            raise ValueError("items1 and items2 must have the same number of rows")
        
        self._impl = self._impl_class(self.n, self.d1, self.d2, items1, items2)
        
        self.data = items1
        self.data_2 = items2
    
    def add(self, items, items2=None):
        if items2 is None:
            self._add_single(items)
        else:
            self._add_dual(items, items2)
            
    def search(self, query, k, lambda_param=0.5, c=1.0, objective="avg"):
        """
        Run DkMIPS algorithm with parameter validation.
        
        Args:
            k (int): Number of items to retrieve
            lambda_param (float): Lambda parameter (default: 0.5)
            c (float): C parameter (default: 1.0)
            query (numpy.ndarray): Query vector with same dimension as items
            objective (str): Objective to use, one of ["avg", "max"]
        
        Returns:
            tuple: (D, I) where D is a list of similarities and I is a list of indices
            
        Raises:
            ValueError: If input parameters are invalid
        """
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        
        if not isinstance(lambda_param, (int, float)) or lambda_param < 0 or lambda_param > 1:
            raise ValueError("lambda_param must be a positive number between 0 and 1")
        
        if not isinstance(c, (int, float)) or c <= 0:
            raise ValueError("c must be a positive number")
        
        if k > self.n:
            raise ValueError(f"k must be less than or equal to the number of items, got {k} and {self.n}")
        
        if query is None:
            raise ValueError("query cannot be None")
            
        if not isinstance(query, np.ndarray):
            query = np.array(query, dtype=np.float32)
        
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        
        if len(query.shape) != 1:
            raise ValueError(f"query must be 1D array, got shape {query.shape}")
        
        if self._is_two_space:
            if not self._is_bc_method:
                objective_map = {
                    "avg": self._impl.dkmips_plus_avg_i2v,
                    "max": self._impl.dkmips_plus_max_i2v,
                }
            else:
                objective_map = {
                    "avg": self._impl.dkmips_avg_i2v,
                    "max": self._impl.dkmips_max_i2v,
                }
        else:
            if not self._is_bc_method:
                objective_map = {
                    "avg": self._impl.dkmips_plus_avg,
                    "max": self._impl.dkmips_plus_max,
                }
            else:
                objective_map = {
                    "avg": self._impl.dkmips_avg,
                    "max": self._impl.dkmips_max,
                }
        
        if objective not in objective_map:
            raise ValueError(f"objective must be one of {list(objective_map.keys())}")
        
        result = objective_map[objective](k, lambda_param, c, query)

        D = []
        I = []
        for i in range(len(result)):
            sim = query @ self.data[result[i]]
            D.append(sim)
            I.append(result[i])

        return D, I
    
class Greedy(DkMIPS):
    """
    A Python wrapper for the C++ Greedy class with additional parameter checking and utilities.
    """
    def __init__(self, d1, d2=None):
        super().__init__(d1, d2, _Greedy, is_bc_method=False)
    
    def __repr__(self):
        return f"Greedy(implementation={self._impl})"

class Dual_Greedy(DkMIPS):
    """
    A Python wrapper for the C++ Dual_Greedy class with additional parameter checking and utilities.
    """
    def __init__(self, d1, d2):
        super().__init__(d1, d2, _Dual_Greedy, is_bc_method=False)
    
    def __repr__(self):
        return f"Dual_Greedy(implementation={self._impl})"
    
class BC_Greedy(DkMIPS):
    def __init__(self, d1, d2):
        super().__init__(d1, d2, _BC_Greedy, is_bc_method=True)
    
    def __repr__(self):
        return f"BC_Greedy(implementation={self._impl})"
    
class BC_Dual(DkMIPS):
    def __init__(self, d1, d2):
        super().__init__(d1, d2, _BC_Dual, is_bc_method=True)
    
    def __repr__(self):
        return f"BC_Dual(implementation={self._impl})"
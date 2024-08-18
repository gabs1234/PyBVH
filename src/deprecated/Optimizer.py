from BVHTree import BVHTree

class OptimizationDelegate(object):
    def __init__(self):
        self._optimization = None
        self._optimizationName = None
        self._optimizationDescription = None
        self._optimizationParameters = None
    
    def optimize(self, tree : BVHTree):
        raise NotImplementedError

class AgglomerativeTreeletOptimizer(OptimizationDelegate):
    def __init__(self):
        super().__init__()
        self._optimizationName = "Agglomerative Treelet Optimization"
        self._optimizationDescription = "Agglomerative Treelet Optimization"
        self._optimizationParameters = None

    def optimize(self, tree : BVHTree) -> None:
        print ("AgglomerativeTreeletOptimizer")

class Optimizer:
    def __init__(self, _OptimizationDelegate : OptimizationDelegate) -> None:
        self._delegate = _OptimizationDelegate

    def optimize(self, tree : BVHTree) -> None:
        self._delegate.optimize(tree)

    def __getattr__(self, name):
        return getattr(self._delegate, name)
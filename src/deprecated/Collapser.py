from BVHTree import BVHTree
from Scene import Scene

class CollapserDelegate(object):
    def __init__(self):
        self._collapser = None
        self._collapserName = None
        self._collapserDescription = None
        self._collapserParameters = None

    def collapse(self, tree : BVHTree) -> BVHTree:
        raise NotImplementedError

class GPUCollapser(CollapserDelegate):
    def __init__(self):
        super().__init__()
        self._collapserName = "GPU Collapser"
        self._collapserDescription = "GPU Collapser"
        self._collapserParameters = None

    def collapse(self, tree : BVHTree) -> None:
        print ("Collapsed for GPU representation")
        return None

class Collapser:
    def __init__(self, _CollapserDelegate : CollapserDelegate):
        self._delegate = _CollapserDelegate

    def collapse(self, tree : BVHTree) -> None:
        self._delegate.collapse(tree)

    def __getattr__(self, name):
        return getattr(self._delegate, name)
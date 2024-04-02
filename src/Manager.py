from BVHTree import BVHTree

class ManagerDelegate(object):
    def __init__(self):
        self._manager = None
        self._managerName = None
        self._managerDescription = None
        self._managerParameters = None
    
    def manage(self, tree : BVHTree) -> None:
        raise NotImplementedError

class CollapsedTreeManager(ManagerDelegate):
    def __init__(self):
        super().__init__()
        self._managerName = "Collapsed Tree Manager"
        self._managerDescription = "Collapsed Tree Manager"
        self._managerParameters = None

    def manage(self, tree : BVHTree) -> None:
        print ("CollapsedTreeManager")
        
class TreeManager(ManagerDelegate):
    def __init__(self):
        super().__init__()
        self._managerName = "Tree Manager"
        self._managerDescription = "Tree Manager"
        self._managerParameters = None

    def manage(self, tree : BVHTree) -> None:
        print ("TreeManager")
    
class Manager:
    def __init__(self, _ManagerDelegate : ManagerDelegate):
        self._delegate = _ManagerDelegate

    def manage(self, tree : BVHTree) -> None:
        self._delegate.manage(tree)

    def __getattr__(self, name):
        return getattr(self._delegate, name)
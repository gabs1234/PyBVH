import numpy as np

IS_LEAF = 2147483648
sorted_keys = [26400869, 620794177, 977888662]

class InternalNodes:
    def __init__(self, N):
        self.child_left = np.zeros(N)
        self.child_right = np.zeros(N)
        self.left_range = np.zeros(N)
        self.right_range = np.zeros(N)
        self.entered = np.zeros(N)

class LeafNodes:
    def __init__(self, keys):
        self.keys = keys

def delta(i, keys):
    return keys[i+1] ^ keys[i]

def updateParent(N, id, inodes, lnodes):

    left_range = id
    right_range = id
    parent  = id
    current_node = id

    is_leaf = True

    while True:
        if (left_range == 0 and right_range == 0):
            print ("Root node")
            break

        if (is_leaf):
            current_node = id | IS_LEAF
        
        if (left_range == 0 or
            (right_range != N and ))
def LVBHApetrei():
    N = len(sorted_keys)

    # Create the leaf nodes
    inodes = InternalNodes(N)
    lnodes = LeafNodes(sorted_keys)

    for i, key in enumerate(sorted_keys):
        updateParent(N, i, inodes, lnodes)
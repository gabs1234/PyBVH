import numpy as np
from collections import deque
import graphviz
from time import sleep
smallest_int32 = np.iinfo(np.int32).min
biggest_int32 = np.iinfo(np.int32).max

UNTOUCHED = -1
SENTINEL = -2

def delta(keys, index):
    if (index < 0 or index >= len(keys) - 1):
        return biggest_int32
    
    a = keys[index]
    b = keys[index + 1]
    x = a ^ b

    return x + (not x) * (smallest_int32 + (index ^ (index + 1)))

left_child = np.ones(8, dtype=np.int32)*-1
ropes_internals = np.zeros(8, dtype=np.int32)
ropes_leafs = np.zeros(8, dtype=np.int32)
ranges = np.ones(8, dtype=np.int32) * UNTOUCHED
keys = [0b00001,0b000010, 0b00100, 0b00101, 0b10011,
            0b11000, 0b11001, 0b11110]

def compareAndSwap(list1, compare, val, index):
    old = list1[index]
    if old == compare:
        list1[index] = val
    
    return old

def buildTree(nb_keys, index):
    range_left = index
    range_right = index 
    delta_left = delta(keys, index-1)
    delta_right = delta(keys, index)

    if (index == nb_keys - 1):
        ropes_leafs[index] = SENTINEL
    else:
        if (delta_right < delta(keys, index + 1)):
            ropes_leafs[index] = index + 1
        else:
            ropes_leafs[index] = index + 1 + nb_keys
    
    i = index
    root = 0 + nb_keys
    q = 0
    while True:
        l = 0
        if delta_right < delta_left:
            p = range_right
            range_right = compareAndSwap(ranges, UNTOUCHED, range_left, p)
            if range_right == UNTOUCHED:
                break
            delta_right = delta(keys, range_left)

            l = i
        else:
            p = range_left - 1
            range_left = compareAndSwap(ranges, UNTOUCHED, range_right, p)
            if range_left == UNTOUCHED:
                break
            delta_left = delta(keys, range_left - 1)

            l = p
            child_left_is_leaf = (l == range_left)

            if (not child_left_is_leaf):
                l = l + nb_keys
            
        if (delta_right < delta_left):
            q = range_right
        else:
            q = range_left
        
        left_child[q] = l
        if range_right == nb_keys - 1:
            ropes_internals[q] = SENTINEL
        else:
            r = range_right + 1
            if delta_right < delta(keys, r):
                ropes_internals[q] = r
            else:
                ropes_internals[q] = r + nb_keys
        
        i = q + nb_keys
        
        if (i == root):
            break

def descendTree():
    root = 0
    queue = deque()
    queue.append([root, 0])

    level_nodes = {}

    while queue:
        v, level = queue.popleft()

        if level not in level_nodes:
            level_nodes[level] = [v]
        else:
            level_nodes[level].append(v)

        # Check if v is leaf
        if v < len(keys):
            rope_list = ropes_leafs
        else: # internal node
            rope_list = ropes_internals
            v -= len(keys)
            left = left_child[v]
        
            queue.append([left, level + 1])

        # Append the skip node
        skip = rope_list[v]
        if skip != SENTINEL:
            queue.append([skip, level + 1])
    
    return level_nodes

def visualizeTree():
    root = 8
    queue = deque()
    queue.append(root)

    set_nodes = set()

    dot = graphviz.Digraph("bvh")
    dot.attr(dpi='300') # Set the DPI to 300

    while set_nodes != len(keys):
        parent_id = queue.popleft()
        parent_label = parent_id

        # get left child
        if parent_id >= len(keys):
            rope_list = ropes_internals
            parent_id -= len(keys)
        else:
            rope_list = ropes_leafs
        
        left = left_child[parent_id]

        if left == -1:
            continue

        left_label = left
        queue.append(left)

        # get right child
        if left > len(keys):
            left -= len(keys)
        
        right = rope_list[left]
        print (right)

        if right == SENTINEL or right in set_nodes:
            continue
        else:
            set_nodes.add(right)

        right_label = right
        queue.append(right_label)

        print ("Parent: ", parent_label, "Left: ", left_label, "Right: ", right_label)
        sleep(1)

        # # visualize
        # dot.node(str(parent_label), str(parent_label))
        # dot.node(str(left_label), str(left_label))
        # dot.node(str(right_label), str(right_label))
        # dot.edge(str(parent_label), str(left_label))
        # dot.edge(str(parent_label), str(right_label))
        
    dot.render('graphviz/bvh_prokepenko.png',format='png')
def main():
    for i in range(len(keys)):
        buildTree(len(keys), i)

    print("Left child: ", left_child)
    print("Ropes internals: ", ropes_internals)
    print("Ropes leafs: ", ropes_leafs)

    visualizeTree()

if __name__ == "__main__":
    main()
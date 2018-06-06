from trees_data_structures import *


def get_optimal_partiotion(data):
    # ToDo
    j = 0
    s = 0
    return j, s


def cart(dataset, max_depth, min_node_size):
    tree = RegressionTree()

    # the i_th list in d consists of tuples of node references and their relevant datasets
    d = [[] for _ in range(max_depth + 1)]
    d[0] = [(dataset.data, tree.get_root())]
    # store all leaves tuples (dataset and reference to node)
    d_leaves = []

    for k in range(max_depth):
        for P, node in d[k]:  # for each depth, iterate over all nodes and split as necessary
            j, s = get_optimal_partiotion(P)
            Pl = P[P[j] <= s]
            Pr = P[P[j] > s]

            # checks minimum node size violation
            if len(Pl) > min_node_size and len(Pr) > min_node_size:
                # initialize descendents
                left_node = RegressionTreeNode()
                right_node = RegressionTreeNode()

                node.left_descendent = left_node
                node.right_descendent = right_node

                # append descendents to the next depth
                d[k+1].append((Pl, left_node))
                d[k+1].append((Pr, right_node))
            else:
                d_leaves.append((P, node))

    # add all nodes in maximum depth to the leaves and calculates each leave value
    d_leaves += d[max_depth]
    for P, node in d_leaves:
        node.const = P[dataset.label_name].mean()

    return tree

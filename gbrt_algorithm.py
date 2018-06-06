from trees_data_structures import *
import numpy as np


def get_optimal_partiotion(data, label_name):
    x = data.drop(label_name)
    y = data[label_name]

    min_so_far = np.inf
    best_col_name, best_split_value = 0, 0

    for col_name in x.columns:
        col_unique_values = x[col_name].unique()
        for uniqe_value in col_unique_values:
            x_left = x[x[col_name] <= uniqe_value]
            x_right = x[x[col_name] > uniqe_value]

            y_left = y[x_left.index]
            y_right = y[x_right.index]

            loss = sum((y_left - y_left.mean())**2) + sum((y_right - y_right.mean())**2)

            if loss < min_so_far:
                min_so_far = loss
                best_col_name = col_name
                best_split_value = uniqe_value

    return best_col_name, best_split_value


def cart(dataset, max_depth, min_node_size):
    tree = RegressionTree()

    # the i_th list in d consists of tuples of node references and their relevant datasets
    tree_levels_list = [[] for _ in range(max_depth + 1)]
    tree_levels_list[0] = [(dataset.data, tree.get_root())]
    # store all leaves tuples (dataset and reference to node)
    leaves = []

    for depth in range(max_depth):
        for node_data, node_reference in tree_levels_list[depth]:  # for each depth, iterate over all nodes and split as necessary
            col_name, split_value = get_optimal_partiotion(node_data, dataset.label_name)
            left_node_data = node_data[node_data[col_name] <= split_value]
            right_node_data = node_data[node_data[col_name] > split_value]

            # checks minimum node size violation
            if len(left_node_data) > min_node_size and len(right_node_data) > min_node_size:
                # initialize descendents
                left_node_reference = RegressionTreeNode()
                right_node_reference = RegressionTreeNode()

                node_reference.left_descendent = left_node_reference
                node_reference.right_descendent = right_node_reference

                # append descendents to the next depth
                tree_levels_list[depth+1].append((left_node_data, left_node_reference))
                tree_levels_list[depth+1].append((right_node_data, right_node_reference))
            else:
                leaves.append((node_data, node_reference))

    # add all nodes in maximum depth to the leaves and calculates each leave value
    leaves += tree_levels_list[max_depth]
    for node_data, node_reference in leaves:
        node_reference.const = node_data[dataset.label_name].mean()

    return tree

import numpy as np


class RegressionTreeNode:
    def __init__(self):
        self.j = None
        self.s = None
        self.left_descendent = None
        self.right_descendent = None
        self.const = None

    def make_terminal(self, const):
        self.const = const

    def split(self, j, s):
        self.j = j
        self.s = s

    def print_sub_tree(self, depth=0):
        identation = '\t' * depth

        if self.is_terminal():
            print('{}return {}'.format(identation, self.const))
        else:
            print("{}if x['{}'] <= {} then:".format(identation, self.j, self.s))
            self.left_descendent.print_sub_tree(depth=depth+1)
            print("if x['{}'] > {} then:".format(self.j, self.s))
            self.right_descendent.print_sub_tree(depth=depth+1)

    def is_terminal(self):
        return self.const is not None

    def evaluate(self, x):
        if self.is_terminal():
            return self.const
        elif x[self.j] <= self.s:
            return self.left_descendent.evaluate(x)
        else:
            return self.right_descendent.evaluate(x)


class RegressionTree:
    def __init__(self):
        self.root = RegressionTreeNode()

    def get_root(self):
        return self.root

    def evaluate(self, x):
        return self.root.evaluate(x)


class RegressionTreeEnsemble:
    def __init__(self):
        self.trees = []
        self.weights = []
        self.M = 0
        self.c = 0

    def add_tree(self, tree, weight):
        self.trees.append(tree)
        self.weights.append(weight)
        self.M += 1

    def set_initial_constant(self, c):
        self.c = c

    def evaluate(self, x, m):
        m = min(m, self.M)

        evals = [tree.evaluate(x)*weight for tree, weight in zip(self.trees[:m], self.weights[:m])]
        
        return self.c + sum(evals)



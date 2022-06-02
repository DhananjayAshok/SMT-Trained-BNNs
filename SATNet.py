import torch
from torch.nn import Linear, Conv2d
import numpy as np

import utils
from utils import CNFBuilder, CNFDebugger
from pysat.solvers import Solver
from pysat.formula import CNF


class SATNet:
    def __init__(self, model):
        """
        model must be sequential
        """
        self.model = model
        self.id_pool = CNFBuilder.get_IDPool()
        self.clauses = []
        self.solution = {}
        self.solver_class = Solver

    def create_linear_params(self, i, layer):
        ids = []
        for j in range(layer.out_features):
            id_list = []
            for iii in range(layer.in_features):
                w = self.id_pool.id(f"Layer[{i}]|w{j, iii}")
                id_list.append(w)
            ids.append(id_list)
        ids = utils.int_array(ids)
        layer.ids = ids

    def update_linear_params(self, i, layer):
        assert self.solution is not None and self.solution != {}
        for j in range(layer.out_features):
            for iii in range(layer.in_features):
                w = self.solution.get(f"Layer[{i}]|w{j, iii}")
                n_w = 1 if w else -1
                layer.weight[j, iii] = n_w

    def create_network_params(self):
        for i in range(len(self.model)):
            layer = self.model[i]
            if isinstance(layer, Linear):
                self.create_linear_params(i, layer)

    def update_network_params(self):
        for i in range(len(self.model)):
            layer = self.model[i]
            if isinstance(layer, Linear):
                self.update_linear_params(i, layer)

    def create_datapoint_params(self, X):
        xs = []
        if len(X.shape) == 2:
            for d in range(X.shape[0]):
                x_list = []
                for iii in range(X.shape[1]):
                    x = self.id_pool.id(f"X{d, iii}")
                    x_list.append(x)
                xs.append(x_list)
            self.clauses.extend(CNFBuilder.assign_values(X, xs, "x"))
        elif len(X.shape) == 3:
            pass
        xs = utils.int_array(xs)
        return xs

    def create_architecture(self, xs):
        hs = None
        n_datapoints = len(xs)
        for i in range(len(self.model)):
            layer = self.model[i]
            if isinstance(layer, Linear):
                clauses = []
                if hs is None:
                    _, ws, xws, hs, clauses = CNFBuilder.create_linear_layer(in_features=layer.in_features,
                                                                             out_features=layer.out_features, xs=xs,
                                                                             n_datapoints=n_datapoints,
                                                                             ws=layer.ids, layer_id=i,
                                                                             id_pool=self.id_pool)
                else:
                    _, ws, xws, hs, clauses = CNFBuilder.create_linear_layer(in_features=layer.in_features,
                                                                             out_features=layer.out_features, xs=hs,
                                                                             n_datapoints=n_datapoints,
                                                                             ws=layer.ids, layer_id=i,
                                                                             id_pool=self.id_pool)
                self.clauses.extend(clauses)
        return hs

    def create_output_params(self, os, Y):
        self.clauses.extend(CNFBuilder.assign_values(inp_array=Y, inp_list=os, val_type="o"))

    def create_sat_model(self, X, Y):
        self.create_network_params()
        xs = self.create_datapoint_params(X)
        os = self.create_architecture(xs)
        self.create_output_params(os, Y)

    def sat_sweep(self, X, y):
        self.create_sat_model(X, y)
        cnf = CNF()
        cnf.extend(self.clauses)
        cnf.vpool = self.id_pool
        solver = self.solver_class(bootstrap_with=cnf)
        status = solver.solve()
        self.solution = CNFDebugger.get_solver_model(solver, cnf)
        print(self.solution)
        if status:
            with torch.no_grad():
                self.update_network_params()
        return

    def forward(self, X):
        h = X
        print("forward")
        for i, layer in enumerate(self.model):
            print(f"Layer: {i}, h: \n{h}")
            print(f"To Be \n{h.matmul(layer.weight.transpose(0, 1))}")
            h = layer(h)
            h = (h > 0).float()
        return h


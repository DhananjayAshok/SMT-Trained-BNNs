import warnings

import torch
from torch.nn import Linear, Conv2d
import numpy as np

import utils
from utils import CNFBuilder, CNFDebugger
from pysat.solvers import Solver
from pysat.formula import CNF, WCNF
from pysat.examples.rc2 import RC2


class SATNet:
    def __init__(self, model):
        """
        model must be sequential
        """
        self.model = model
        self.solver_class = Solver
        self.id_pool = CNFBuilder.get_IDPool()
        self.hard_clauses = []
        self.soft_clauses = []
        self.solution = {}
        self.reset()

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

    def get_linear_param_ids(self, i, out_features, in_features):
        """
        Gives back the linear weight params ids as in the CNF for layer i
        """
        ws = []
        for j in range(out_features):
            w = []
            for iii in range(in_features):
                w.append(self.id_pool.id(f"Layer[{i}]|w{j, iii}"))
            ws.append(w)
        return ws

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
            self.hard_clauses.extend(CNFBuilder.assign_values(X, xs, "x"))
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
                self.hard_clauses.extend(clauses)
        return hs

    def create_output_params(self, os, Y):
        self.soft_clauses.extend(CNFBuilder.assign_values(inp_array=Y, inp_list=os, val_type="o"))

    def create_sat_model(self, X, Y):
        self.create_network_params()
        xs = self.create_datapoint_params(X)
        os = self.create_architecture(xs)
        self.create_output_params(os, Y)

    def sat_sweep(self, X, y):
        self.create_sat_model(X, y)
        cnf = CNF()
        cnf.extend(self.hard_clauses)
        cnf.extend(self.soft_clauses)
        cnf.vpool = self.id_pool
        solver = self.solver_class(bootstrap_with=cnf)
        status = solver.solve()
        self.solution = CNFDebugger.get_solver_model(solver, cnf)
        # print(self.solution)
        if status:
            with torch.no_grad():
                self.update_network_params()
        else:
            warnings.warn("SAT Solver Found No Solution")

    def max_sat_sweep(self, X, y):
        self.create_sat_model(X, y)
        cnf = WCNF()
        cnf.extend(self.hard_clauses)
        cnf.extend(self.soft_clauses, weights=[1 for _ in self.soft_clauses])
        cnf.vpool = self.id_pool
        with RC2(cnf) as rc2:
            for i, m in enumerate(rc2.enumerate()):
                print('model {0} has cost {1}'.format(i, rc2.cost))
                if i >= 5:
                    break
        return

    def reset(self):
        self.id_pool = CNFBuilder.get_IDPool()
        self.hard_clauses = []
        self.soft_clauses = []
        self.solution = {}

    def forward(self, X):
        h = X
        for i, layer in enumerate(self.model):
            h = layer(h)
            h = 2*(h > 0).float()-1
        return h


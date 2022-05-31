import pytest
from SATNet import SATNet
import torch.nn as nn
import torch
from pysat.formula import CNF
from utils import CNFDebugger


def single_layer_network(in_dim, out_dim):
    layer = nn.Linear(in_features=in_dim, out_features=out_dim)
    model = nn.Sequential(layer)
    return SATNet(model)


def double_layer_network(in_dim, mid_dim, out_dim):
    layer = nn.Linear(in_features=in_dim, out_features=mid_dim)
    layer1 = nn.Linear(in_features=mid_dim, out_features=out_dim)
    model = nn.Sequential(layer, layer1)
    return SATNet(model)


def rand_binary(*shape):
    arr = torch.rand(shape)
    return arr > 0


def test_create_linear_params():
    in_dim = 3
    out_dim = 2
    satnet = single_layer_network(in_dim, out_dim)
    layer = satnet.model[0]
    satnet.create_linear_params(0, layer)
    print(layer.ids)
    ids = [[1, 2, 3], [4, 5, 6]]
    print(layer.ids, type(layer.ids))
    assert layer.ids == ids
    assert satnet.id_pool.obj(1) == f"Layer[{0}]|w{0, 0}"
    assert satnet.id_pool.obj(2) == f"Layer[{0}]|w{0, 1}"
    assert satnet.id_pool.obj(3) == f"Layer[{0}]|w{0, 2}"
    assert satnet.id_pool.obj(4) == f"Layer[{0}]|w{1, 0}"
    assert satnet.id_pool.obj(5) == f"Layer[{0}]|w{1, 1}"
    assert satnet.id_pool.obj(6) == f"Layer[{0}]|w{1, 2}"


def test_create_network_params():
    in_dim = 3
    mid_dim = 2
    out_dim = 1
    satnet = double_layer_network(in_dim=in_dim, mid_dim=mid_dim, out_dim=out_dim)
    satnet.create_network_params()
    l0 = satnet.model[0]
    l1 = satnet.model[1]
    id0 = [[1, 2, 3], [4, 5, 6]]
    id1 = [[7, 8]]
    assert l0.ids == id0
    assert l1.ids == id1


def test_create_datapoint_params():
    # Unsure how to test yet
    in_dim = 3
    mid_dim = 2
    out_dim = 1
    satnet = double_layer_network(in_dim=in_dim, mid_dim=mid_dim, out_dim=out_dim)
    satnet.create_network_params()
    X = rand_binary(32, in_dim)
    satnet.create_datapoint_params(X)
    print(satnet.clauses)
    print(CNFDebugger.clause_translate(satnet.clauses, satnet.id_pool))
    cnf = CNF()
    cnf.extend(satnet.clauses)
    cnf.vpool= satnet.id_pool
    solver = satnet.solver_class(bootstrap_with=cnf)
    solver.solve()
    print("Core ", solver.get_core())
    solution = CNFDebugger.get_solver_model(solver, cnf)
    assert solution is not None


def test_create_architecture():
    # Unsure how to test yet
    in_dim = 3
    mid_dim = 2
    out_dim = 1
    satnet = double_layer_network(in_dim=in_dim, mid_dim=mid_dim, out_dim=out_dim)
    satnet.create_network_params()
    X = rand_binary(32, in_dim)
    xs = satnet.create_datapoint_params(X)
    os = satnet.create_architecture(xs)
    print(CNFDebugger.clause_translate(satnet.clauses, satnet.id_pool))
    cnf = CNF()
    cnf.extend(satnet.clauses)
    cnf.vpool= satnet.id_pool
    solver = satnet.solver_class(bootstrap_with=cnf)
    solver.solve()
    print("Core ", solver.get_core())
    solution = CNFDebugger.get_solver_model(solver, cnf)
    assert solution is not None
    assert False


def test_create_output_params():
    # Unsure how to test yet
    in_dim = 3
    mid_dim = 2
    out_dim = 1
    satnet = double_layer_network(in_dim=in_dim, mid_dim=mid_dim, out_dim=out_dim)
    satnet.create_network_params()
    X = rand_binary(32, in_dim)
    y = rand_binary(32, out_dim)
    xs = satnet.create_datapoint_params(X)
    assert len(xs) == 32
    assert len(xs[0]) == in_dim
    os = satnet.create_architecture(xs)
    assert len(os) == 32
    assert len(os[0]) == out_dim
    satnet.create_output_params(os, y)
    cnf = CNF()
    cnf.extend(satnet.clauses)
    cnf.vpool= satnet.id_pool
    solver = satnet.solver_class(bootstrap_with=cnf)
    solver.solve()
    print("Core ", solver.get_core())
    solution = CNFDebugger.get_solver_model(solver, cnf)
    assert solution is not None
    assert True


def test_sat_sweep():
    in_dim = 3
    mid_dim = 2
    out_dim = 1
    satnet = double_layer_network(in_dim=in_dim, mid_dim=mid_dim, out_dim=out_dim)
    satnet.create_network_params()
    X = rand_binary(32, in_dim)
    y = rand_binary(32, out_dim)
    satnet.sat_sweep(X, y)
    assert True
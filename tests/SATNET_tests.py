import pytest
from SATNet import SATNet
import torch.nn as nn
import torch
from pysat.formula import CNF
from utils import CNFDebugger, CNFBuilder


def single_layer_network(in_dim, out_dim):
    layer = nn.Linear(in_features=in_dim, out_features=out_dim, bias=False)
    model = nn.Sequential(layer)
    return SATNet(model)


def double_layer_network(in_dim, mid_dim, out_dim):
    layer = nn.Linear(in_features=in_dim, out_features=mid_dim, bias=False)
    layer1 = nn.Linear(in_features=mid_dim, out_features=out_dim, bias=False)
    model = nn.Sequential(layer, layer1)
    return SATNet(model)


def rand_binary(*shape, minus=True):
    arr = torch.rand(shape)
    if not minus:
        return (arr > 0.5).float()
    else:
        return 2*((arr > 0.5).float()) - 1


def rand_X_y(in_dim, out_dim, bs=32):
    return rand_binary(bs, in_dim), rand_binary(bs, out_dim)


def X_halfsum(in_dim, bs=32):
    X = rand_binary(bs, in_dim)
    y = (2*(X.sum(axis=-1) > 0).float()-1).reshape(bs, 1)
    return X, y


def parity(in_dim=4, bs=32):
    X = rand_binary(bs, in_dim)
    y = (2*(((X/2+0.5).sum(axis=-1) % 2 == 1).float())-1).reshape(bs, 1)
    return X, y


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
    # print(satnet.clauses)
    # print(CNFDebugger.clause_translate(satnet.clauses, satnet.id_pool))
    cnf = CNF()
    cnf.extend(satnet.hard_clauses)
    cnf.vpool= satnet.id_pool
    solver = satnet.solver_class(bootstrap_with=cnf)
    solver.solve()
    solution = CNFDebugger.get_solver_model(solver, cnf)
    assert solution is not None
    X = X/2 + 0.5
    for d in range(len(X)):
        for i in range(in_dim):
            assert solution[f"X{d, i}"] == X[d, i]


def test_create_architecture_1_layer():
    # bs, in_dim, out_dim
    configurations = [(3, 3, 2), (5, 2, 3), (1, 1, 1), (1, 2, 1), (1, 1, 3)]
    for bs, in_dim, out_dim in configurations:
        #  print("_"*10)
        #  print(f"Configurations: {bs, in_dim, out_dim}")
        satnet = single_layer_network(in_dim, out_dim)
        satnet.create_network_params()
        X = rand_binary(bs, in_dim)
        ws_array = rand_binary(out_dim, in_dim)
        ws_inds = satnet.get_linear_param_ids(i=0, out_features=out_dim, in_features=in_dim)
        expected_o = (X.matmul(ws_array.transpose(0, 1)) > 0)
        xs = satnet.create_datapoint_params(X)
        os = satnet.create_architecture(xs)
        cnf = CNF()
        cnf.extend(satnet.hard_clauses)
        clauses = CNFBuilder.assign_values(ws_array, ws_inds, val_type="w")
        cnf.extend(clauses)
        cnf.vpool = satnet.id_pool
        solver = satnet.solver_class(bootstrap_with=cnf)
        solver.solve()
        solution = CNFDebugger.get_solver_model(solver, cnf)
        assert solution is not None
        """
        print(f"\nX")
        print(X)
        print("W")
        print(ws_array)
        print("XW^T")
        print(X.matmul(ws_array.transpose(0, 1)))
        """
        for d in range(bs):
            for m in range(out_dim):
                #  print(f"Checking {satnet.id_pool.obj(os[d][m])} {solution[satnet.id_pool.obj(os[d][m])]} vs "
                #      f"{expected_o[d, m]}")
                assert solution[satnet.id_pool.obj(os[d][m])] == expected_o[d, m]


def test_create_architecture_2_layer():
    # bs, in_dim, mid_dim, out_dim
    configurations = [(3, 3, 2, 2), (5, 2, 1, 3), (1, 1, 1,  1), (1, 2, 3, 1), (1, 1, 2, 3)]
    for bs, in_dim, mid_dim, out_dim in configurations:
        #  print("_"*10)
        #  print(f"Configurations: {bs, in_dim, out_dim}")
        satnet = double_layer_network(in_dim, mid_dim, out_dim)
        satnet.create_network_params()
        X = rand_binary(bs, in_dim)
        ws_array0 = rand_binary(mid_dim, in_dim)
        ws_inds0 = satnet.get_linear_param_ids(i=0, out_features=mid_dim, in_features=in_dim)
        expected_h_bool = X.matmul(ws_array0.transpose(0, 1)) > 0
        expected_h = 2 * expected_h_bool.float() - 1
        ws_array1 = rand_binary(out_dim, mid_dim)
        ws_inds1 = satnet.get_linear_param_ids(i=1, out_features=out_dim, in_features=mid_dim)
        expected_o_bool = expected_h.matmul(ws_array1.transpose(0, 1)) > 0

        xs = satnet.create_datapoint_params(X)
        os = satnet.create_architecture(xs)
        cnf = CNF()
        cnf.extend(satnet.hard_clauses)
        clauses = CNFBuilder.assign_values(ws_array0, ws_inds0, val_type="w")
        cnf.extend(clauses)
        clauses = CNFBuilder.assign_values(ws_array1, ws_inds1, val_type="w")
        cnf.extend(clauses)
        cnf.vpool = satnet.id_pool
        solver = satnet.solver_class(bootstrap_with=cnf)
        solver.solve()
        solution = CNFDebugger.get_solver_model(solver, cnf)
        assert solution is not None
        """
        print(f"\nX")
        print(X)
        print("W")
        print(ws_array)
        print("XW^T")
        print(X.matmul(ws_array.transpose(0, 1)))
        """
        for d in range(bs):
            for m in range(out_dim):
                #  print(f"Checking {satnet.id_pool.obj(os[d][m])} {solution[satnet.id_pool.obj(os[d][m])]} vs "
                #      f"{expected_o[d, m]}")
                assert solution[satnet.id_pool.obj(os[d][m])] == expected_o_bool[d, m]


def test_create_output_params():
    in_dim = 3
    out_dim = 1
    bs = 2
    satnet = single_layer_network(in_dim, out_dim)
    satnet.create_network_params()
    X, y = X_halfsum(in_dim=in_dim, bs=bs)
    xs = satnet.create_datapoint_params(X)
    assert len(xs) == bs
    assert len(xs[0]) == in_dim
    os = satnet.create_architecture(xs)
    assert len(os) == bs
    assert len(os[0]) == out_dim
    satnet.create_output_params(os, y)
    cnf = CNF()
    cnf.extend(satnet.hard_clauses)
    cnf.extend(satnet.soft_clauses)
    cnf.vpool= satnet.id_pool
    solver = satnet.solver_class(bootstrap_with=cnf)
    solver.solve()
    solution = CNFDebugger.get_solver_model(solver, cnf)
    expected_o = y == 1
    assert solution is not None
    print(solution)
    for d in range(bs):
        for m in range(out_dim):
            #  print(f"Checking {satnet.id_pool.obj(os[d][m])} {solution[satnet.id_pool.obj(os[d][m])]} vs "
            #      f"{expected_o[d, m]}")
            assert solution[satnet.id_pool.obj(os[d][m])] == expected_o[d, m]


def test_sat_sweep_1_layer():
    # bs, in_dim, out_dim
    configurations = [(3, 3, 2), (5, 2, 3), (1, 1, 1), (1, 2, 1), (1, 1, 3)]
    for bs, in_dim, out_dim in configurations:
        satnet = single_layer_network(in_dim=in_dim, out_dim=out_dim)
        satnet.create_network_params()
        X = rand_binary(bs, in_dim)
        W = rand_binary(out_dim, in_dim)
        y = 2 * (X.matmul(W.transpose(0, 1)) > 0).float() - 1
        satnet.sat_sweep(X, y)
        pred = satnet.forward(X)
        a = torch.rand(size=(bs, 2 * out_dim))
        a[:, :out_dim] = y
        a[:, out_dim:] = pred
        assert (y == pred).all()


def test_sat_sweep_2_layer():
    # bs, in_dim, mid_dim, out_dim
    configurations = [(3, 3, 2, 2), (5, 2, 1, 3), (1, 1, 1, 1), (1, 2, 3, 1), (1, 1, 2, 3)]
    for bs, in_dim, mid_dim, out_dim in configurations:
        satnet = double_layer_network(in_dim=in_dim, mid_dim=mid_dim, out_dim=out_dim)
        satnet.create_network_params()
        X = rand_binary(bs, in_dim)
        W0 = rand_binary(mid_dim, in_dim)
        W1 = rand_binary(out_dim, mid_dim)
        h = 2 * (X.matmul(W0.transpose(0, 1)) > 0).float() - 1
        y = 2 * (h.matmul(W1.transpose(0, 1)) > 0).float() - 1
        satnet.sat_sweep(X, y)
        pred = satnet.forward(X)
        a = torch.rand(size=(bs, 2 * out_dim))
        a[:, :out_dim] = y
        a[:, out_dim:] = pred
        assert (y==pred).all()


def test_parity_sat():
    satnet = double_layer_network(in_dim=4, mid_dim=2, out_dim=1)
    satnet.create_network_params()
    X, y = parity(4, 32)
    satnet.sat_sweep(X, y)
    pred = satnet.forward(X)
    a = (y - pred)
    assert all(a==0)
    print(f"Perfect on Training Set")
    X, y = parity(4, 32)
    pred = satnet.forward(X)
    a = (y - pred)
    assert all(a == 0)


def test_parity_maxsat():
    satnet = double_layer_network(in_dim=4, mid_dim=2, out_dim=1)
    satnet.create_network_params()
    X, y = parity(4, 32)
    satnet.max_sat_sweep(X, y)
    pred = satnet.forward(X)
    a = (y - pred)
    assert all(a == 0)
    print(f"Perfect on Training Set")
    X, y = parity(4, 32)
    pred = satnet.forward(X)
    a = (y - pred)
    assert all(a == 0)

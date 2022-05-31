import pytest
from utils import CNFBuilder, IDPool, CNFDebugger
from pysat.solvers import Solver
from pysat.formula import CNF
import numpy as np


def basic_setup():
    """
    Gives back a CNF with 4 variables. 1 is set to true, 2 is set to false, 3 and 4 are unset.
    """
    a, b, c, d = 1, 2, 3, 4
    vpool = IDPool()
    a = vpool.id("a")
    b = vpool.id("b")
    c = vpool.id("c")
    d = vpool.id("d")
    cnf = CNF()
    cnf.append([a])
    cnf.append([-b])
    cnf.vpool = vpool
    return cnf, a, b, c, d


def basic_solver(cnf):
    s = Solver(bootstrap_with=cnf)
    status = s.solve()
    if status:
        return CNFDebugger.get_solver_model(s, cnf)
    return status


def basic_dataset():
    n_datapoints = 10
    in_dim=3
    X = np.random.rand(n_datapoints, in_dim) > 0.5
    y = X.sum(axis=1) > in_dim//2
    return X, y


def test_get_IDPool():
    id_pool = CNFBuilder.get_IDPool()
    assert type(id_pool) == IDPool


def test_implies():
    basic_cnf, a, b, c, d = basic_setup()
    # First test 1 => 3 and 2 => 4
    oit = CNFBuilder.implies(a, c)
    tif = CNFBuilder.implies(b, d)
    basic_cnf.extend([oit, tif])
    status = basic_solver(basic_cnf)
    assert status is not False
    assert status["c"]

    basic_cnf, a, b, c, d = basic_setup()
    # Next test 1 => 3 and -3 has no solutions
    oit = CNFBuilder.implies(a, c)
    c2 = [-c]
    basic_cnf.extend([oit, c2])
    status = basic_solver(basic_cnf)
    assert not status

    basic_cnf, a, b, c, d = basic_setup()
    # Finally 2 => 4 and 4 can be either true or false
    oit = [d]
    tif = CNFBuilder.implies(b, d)
    basic_cnf.extend([oit, tif])
    status = basic_solver(basic_cnf)
    assert status is not False

    # Finally 2 => 4 and 4 can be either true or false
    basic_cnf, a, b, c, d = basic_setup()
    oit = [-d]
    tif = CNFBuilder.implies(b, d)
    basic_cnf.extend([oit, tif])
    status = basic_solver(basic_cnf)
    assert status is not False


def test_iff():
    basic_cnf, a, b, c, d = basic_setup()
    # First test 1 <=> 3 and 2 <=> 4
    oit = CNFBuilder.iff(a, c)
    tif = CNFBuilder.iff(b, d)
    basic_cnf.extend(oit)
    basic_cnf.extend(tif)
    status = basic_solver(basic_cnf)
    assert status is not False
    assert status["c"]
    assert not status["d"]


def test_aiffboc():
    # testing a <=> b or c
    # First test True <=> False or ? makes ? into True
    basic_cnf, a, b, c, d = basic_setup()
    clauses = CNFBuilder.aiffboc(a, b, c)
    basic_cnf.extend(clauses)
    status = basic_solver(basic_cnf)
    assert status is not False
    assert status["c"]

    # Next test False <=> ? or ? makes both into False
    basic_cnf, a, b, c, d = basic_setup()
    clauses = CNFBuilder.aiffboc(b, c, d)
    basic_cnf.extend(clauses)
    status = basic_solver(basic_cnf)
    assert status is not False
    assert not status["c"]
    assert not status["d"]

    # Test True <=> False or False is no solution
    basic_cnf, a, b, c, d = basic_setup()
    clauses = CNFBuilder.aiffboc(a, b, c)
    basic_cnf.extend(clauses)
    basic_cnf.append([-c])
    status = basic_solver(basic_cnf)
    assert not status

    # Test False <=> True or ? is no solution
    basic_cnf, a, b, c, d = basic_setup()
    clauses = CNFBuilder.aiffboc(b, a, d)
    basic_cnf.extend(clauses)
    status = basic_solver(basic_cnf)
    assert not status


def test_aiffbacod():
    # Testing A <=> (B and C) or D
    # Test True <=> (False and ?) or ? sets D to True
    basic_cnf, a, b, c, d = basic_setup()
    clauses = CNFBuilder.aiffbacod(a, b, c, d)
    basic_cnf.extend(clauses)
    status = basic_solver(basic_cnf)
    assert status is not False
    assert status["d"]

    # Test False <=> (True and ?) or ? Sets both to False
    basic_cnf, a, b, c, d = basic_setup()
    clauses = CNFBuilder.aiffbacod(b, a, c, d)
    basic_cnf.extend(clauses)
    status = basic_solver(basic_cnf)
    assert status is not False
    assert not status["d"]
    assert not status["c"]

    # Test True <=> (False and ?) or False has no solution
    basic_cnf, a, b, c, d = basic_setup()
    clauses = CNFBuilder.aiffbacod(a, b, c, d)
    basic_cnf.extend(clauses)
    basic_cnf.append([-d])
    status = basic_solver(basic_cnf)
    assert not status


def test_sequential_counter():
    # Should add more tests for correctness
    prefix = "test"
    # Test when sum = 1
    basic_cnf, a, b, c, d = basic_setup()
    vpool = basic_cnf.vpool
    basic_cnf.append([-c])
    basic_cnf.append([-d])
    clauses, _, final = CNFBuilder.sequential_counter([a, b, c, d], vpool=vpool, prefix=prefix, C=4)
    basic_cnf.extend(clauses)
    status = basic_solver(basic_cnf)
    assert status is not False
    for i in range(1, 4+1):
        assert status[f"{prefix}_r{i, 1}"]
    for i in range(1, 4+1):
        assert not status[f"{prefix}_r{i, 2}"]
        assert not status[f"{prefix}_r{i, 3}"]
        assert not status[f"{prefix}_r{i, 4}"]

    # Test when sum = 2
    basic_cnf, a, b, c, d = basic_setup()
    vpool = basic_cnf.vpool
    basic_cnf.append([c])
    basic_cnf.append([-d])
    clauses, _, final = CNFBuilder.sequential_counter([a, b, c, d], vpool=vpool, prefix=prefix, C=4)
    basic_cnf.extend(clauses)
    status = basic_solver(basic_cnf)
    assert status is not False
    for i in range(1, 4 + 1):
        assert status[f"{prefix}_r{i, 1}"]
    for i in range(1, 4 + 1):
        assert not status[f"{prefix}_r{i, 3}"]
        assert not status[f"{prefix}_r{i, 4}"]
    assert status[f"{prefix}_r{3, 2}"]
    assert status[f"{prefix}_r{4, 2}"]

    # Test when sum = 3
    basic_cnf, a, b, c, d = basic_setup()
    vpool = basic_cnf.vpool
    basic_cnf.append([c])
    basic_cnf.append([d])
    clauses, _, final = CNFBuilder.sequential_counter([a, b, c, d], vpool=vpool, prefix=prefix, C=4)
    basic_cnf.extend(clauses)
    status = basic_solver(basic_cnf)
    assert status is not False
    for i in range(1, 4 + 1):
        assert status[f"{prefix}_r{i, 1}"]
    for i in range(1, 4 + 1):
        assert not status[f"{prefix}_r{i, 4}"]
        assert not status[f"{prefix}_r{i, 4}"]
    assert status[f"{prefix}_r{3, 2}"]
    assert status[f"{prefix}_r{4, 2}"]
    assert status[f"{prefix}_r{4, 3}"]


def test_xnor_link():
    # Test full truth table
    basic_cnf, a, b, c, d = basic_setup()
    e = basic_cnf.vpool.id("e")
    f = basic_cnf.vpool.id("f")
    g = basic_cnf.vpool.id("g")
    h = basic_cnf.vpool.id("h")
    basic_cnf.extend([[c], [-d]])
    basic_cnf.extend(CNFBuilder.xnor_link(b, d, e))
    basic_cnf.extend(CNFBuilder.xnor_link(b, a, f))
    basic_cnf.extend(CNFBuilder.xnor_link(a, b, g))
    basic_cnf.extend(CNFBuilder.xnor_link(a, c, h))
    status = basic_solver(basic_cnf)
    assert status
    assert status["e"]
    assert not status["f"]
    assert not status["g"]
    assert status["h"]


def test_linear_product_link():
    basic_cnf, a, b, c, d = basic_setup()
    e = basic_cnf.vpool.id("e")
    f = basic_cnf.vpool.id("f")
    g = basic_cnf.vpool.id("g")
    h = basic_cnf.vpool.id("h")
    i = basic_cnf.vpool.id("i")
    j = basic_cnf.vpool.id("j")
    xs = [[a, b]]
    ws = [[c, d], [e, f]]
    xws = [[[g, h], [i, j]]]
    clauses = [[c], [d], [-e], [-f]]
    # We should have g = ac, h = bd, i = ae, j = bf where multiplication is XNOR
    basic_cnf.extend(clauses)
    clauses = CNFBuilder.linear_product_link(xs, ws, xws)
    basic_cnf.extend(clauses)
    status = basic_solver(basic_cnf)
    print(status)
    assert status is not False
    assert status["g"]
    assert not status["h"]
    assert not status["i"]
    assert status["j"]


def test_create_layer():
    basic_cnf, a, b, c, d = basic_setup()
    a1 = basic_cnf.vpool.id("a1")
    b1 = basic_cnf.vpool.id("b1")
    e = basic_cnf.vpool.id("e")
    f = basic_cnf.vpool.id("f")
    in_features = 3
    out_features = 2
    id_pool = basic_cnf.vpool
    do_linking = False
    ws = [[a1, b1, c], [d, e, f]]
    xs, ws, xws, hs, clauses = CNFBuilder.create_linear_layer(in_features, out_features, n_datapoints=1, ws=ws,
                                                              id_pool=id_pool, layer_id="0", xs=[],
                                                              do_linking=do_linking)
    clauses.extend([[xs[0][0]], [-xs[0][1]]])
    for i in range(2):
        clauses.append([ws[0][i]])
        clauses.append([-ws[1][i]])

    basic_cnf.extend(clauses)
    status = basic_solver(basic_cnf)
    assert len(xs) == 1
    assert len(xs[0]) == in_features
    assert len(xws) == 1
    assert len(xws) == 1
    assert len(xws[0]) == out_features
    assert len(xws[0][0]) == in_features
    assert len(hs) == 1
    assert len(hs[0]) == out_features
    print(status)
    assert status is not False
    assert status[f"l{0},d{0},xw{0, 0}"]
    assert not status[f"l{0},d{0},xw{0, 1}"]
    assert not status[f"l{0},d{0},xw{1, 0}"]
    assert status[f"l{0},d{0},xw{1, 1}"]


def test_nn():
    return
    X, Y = basic_dataset()
    # nn architecture: 3 -> 2 -> 1
    id_pool = CNFBuilder.get_IDPool()
    all_ws = []
    all_xs = []
    os = []
    clauses = []
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        xs, ws, xws, c = CNFBuilder.create_layer(3, 2, id_pool=id_pool, layer_id="0",
                                                       datapoint_id=f"{i}",
                                                       create_input=True, xs=[], do_linking=True)
        all_ws.append(ws)
        clauses.extend(c)
        all_xs.append(xs)
        hs = []
        for j in range(2):
            h_clauses, _, h = CNFBuilder.sequential_counter(xws[j], vpool=id_pool, prefix=f"h_{j}_d_{i}", C=2)
            clauses.extend(h_clauses)
            hs.append(h)
        hs, h_ws, hws, o_clauses = CNFBuilder.create_layer(2, 1, id_pool=id_pool, layer_id="1",
                                                       datapoint_id=f"{i}",
                                                       create_input=False, xs=hs, do_linking=True)
        all_ws.append(h_ws)
        clauses.extend(o_clauses)
        output_clauses, _, o = CNFBuilder.sequential_counter(hws[0], vpool=id_pool, prefix=f"o{i}", C=1)
        os.append(o)
        clauses.extend(output_clauses)
    clauses.extend(CNFBuilder.assign_values(inp_array=X, inp_list=all_xs, val_type="x"))
    clauses.extend(CNFBuilder.assign_values(inp_array=Y, inp_list=os, val_type="o"))
    cnf = CNF()
    cnf.vpool = id_pool
    cnf.extend(clauses)
    status = basic_solver(cnf)
    print(status)
    assert not status




















import aiger
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver


def implies(x, y):
    return [-x, y]


def iff(x, y):
    return implies(x, y), implies(y, x)


def aiffboc(a, b, c):
    """
    Returns clauses for A <=> B | C
    """
    aiboc = [-a, b, c]
    bocia0 = [-b, a]
    bocia1 = [-c, a]
    return aiboc, bocia0, bocia1


def aiffbacod(a, b, c, d):
    """
    Returns clauses for A <=> (B and C) or D
    """
    aibacod0 = [-a, b, d]
    aibacod1 = [-a, c, d]
    bacodia0 = [-b, -c, a]
    bacodia1 = [-d, a]
    return aibacod0, aibacod1, bacodia0, bacodia1


def cnf_int_to_var_name(var, vpool):
    item = vpool.obj(abs(var))
    return item


def print_cnf_vars(cnf):
    vars = []
    for inp in range(1, cnf.nv+1):
        name = cnf_int_to_var_name(inp, cnf.vpool)
        if name is not None:
            vars.append(f"{inp, name}")
    print(vars)
    return vars


def clause_translate(clauses, vpool):
    translated = []
    for clause in clauses:
        terms = []
        for term in clause:
            name = cnf_int_to_var_name(term, vpool)
            if name is None:
                name = "?"
            if term < 0:
                name = f"~{name}"
            terms.append(name)
        translated.append(terms)
    return translated


def get_solver_model(solver, cnf):
    model = solver.get_model()
    m = {}
    for var in model:
        name = cnf_int_to_var_name(var, cnf.vpool)
        if name is not None:
            m[name] = var > 0
    return m


def sequential_counter(var_list, vpool=None, prefix="", C=None):
    """
    var_list is a list of integers as per pysat CNF
    vpool is an IDPool with all variables stored till now, or a fresh one is created
    prefix is a string identifier used to distinguish between multiple scs in the same CNF
    If C is none we assume C = len(var_list) = m
    """
    clauses = []
    if vpool is None:
        vpool = IDPool()
    r = {}
    m = len(var_list)
    if C is None:
        C = m
    for i in range(1, m + 1):
        for j in range(1, C+1):
            r[(i, j)] = vpool.id(f"{prefix}_r{i, j}")
    if m == 0 or C <= 0:
        raise ValueError(f"Yo What")
    clauses.extend(iff(var_list[0], r[1, 1]))
    for j in range(2, C+1):
        clauses.append([-r[1, j]])
    for i in range(2, m+1):
        clauses.extend(aiffboc(r[i, 1], var_list[i-1], r[i-1, 1]))
    for i in range(2, m+1):
        for j in range(2, C+1):
            clauses.extend(aiffbacod(r[i, j], var_list[i-1], r[i-1, j-1], r[i-1, j]))
    return clauses, vpool, r[m, C]


vpool = IDPool()
x = vpool.id(f"x")
y = vpool.id(f"y")
z = vpool.id(f"z")
sc, vp, r = sequential_counter([x, y, z], vpool=vpool, C=2)
f = CNF()
f.extend(sc)
f.vpool = vpool
print_cnf_vars(f)


def proc(cnf):
    s = Solver(bootstrap_with=cnf)
    status = s.solve()
    if status:
        print(get_solver_model(s, cnf))
    else:
        print(f"No Solution Found")


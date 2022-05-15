import aiger
from pysat.formula import CNF
from pysat.solvers import Solver


def iff(x, y):
    xiffy = (~x | y) & (~y | x)
    return xiffy


def cnf_int_to_var_name(var, cnf):
    item = cnf.vpool.obj(abs(var))
    if hasattr(item, 'name'):
        return item.name
    else:
        return None


def print_cnf_vars(cnf):
    vars = []
    for inp in cnf.inps:
        name = cnf_int_to_var_name(inp, cnf)
        if name is not None:
            vars.append(f"{inp, name}")
    print(vars)
    return vars


def get_solver_model(solver, cnf):
    model = solver.get_model()
    m = {}
    for var in model:
        name = cnf_int_to_var_name(var, cnf)
        if name is not None:
            m[name] = var > 0
    return m


def sequential_counter(var_list, C=None):
    """
    If C is none we assume C = len(var_list) = m
    """
    r = {}
    m = len(var_list)
    if C is None:
        C = m
    for i in range(1, m + 1):
        for j in range(1, C+1):
            r[(i, j)] = aiger.atom(f"r{i, j}")
    if m == 0 or C <= 0:
        raise ValueError(f"Yo What")
    clause_set1 = iff(var_list[0], r[1, 1])
    return clause_set1, r
    for j in range(2, C+1):
        clause_set1 = clause_set1 & ~r[1, j]
    expressions = clause_set1
    for i in range(2, m+1):
        expressions = expressions & iff(r[i, 1], var_list[i-1] | r[i-1, 1])
    for i in range(2, m+1):
        for j in range(2, C+1):
            extension = iff(r[i, j], (var_list[i-1] & r[i-1, j-1]) | r[i-1, j])
            expressions = expressions & extension
    return expressions, r[m, C]


x, y, z = aiger.atoms("x", "y", "z")
sc, r = sequential_counter([x, y, z], 2)
f = CNF(from_aiger=sc)
print_cnf_vars(f)
s = Solver(bootstrap_with=f)


def proc(sol, cnf):
    sol.solve()
    print(get_solver_model(sol, cnf))


from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
from utils import CNFBuilder, CNFDebugger




vpool = IDPool()
x = vpool.id(f"x")
y = vpool.id(f"y")
z = vpool.id(f"z")
sc, vp, r = CNFBuilder.sequential_counter([x, y, z], vpool=vpool, C=2)
f = CNF()
f.extend(sc)
f.vpool = vpool
CNFDebugger.print_cnf_vars(f)


def proc(cnf):
    s = Solver(bootstrap_with=cnf)
    status = s.solve()
    if status:
        print(CNFDebugger.get_solver_model(s, cnf))
    else:
        print(f"No Solution Found")
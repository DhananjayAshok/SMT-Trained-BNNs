from pysat.formula import CNF, IDPool
from pysat.solvers import Solver


class CNFBuilder:
    @staticmethod
    def implies(x, y):
        return [-x, y]

    @staticmethod
    def iff(x, y):
        return CNFBuilder.implies(x, y), CNFBuilder.implies(y, x)

    @staticmethod
    def aiffboc(a, b, c):
        """
        Returns clauses for A <=> B | C
        """
        aiboc = [-a, b, c]
        bocia0 = [-b, a]
        bocia1 = [-c, a]
        return aiboc, bocia0, bocia1

    @staticmethod
    def aiffbacod(a, b, c, d):
        """
        Returns clauses for A <=> (B and C) or D
        """
        aibacod0 = [-a, b, d]
        aibacod1 = [-a, c, d]
        bacodia0 = [-b, -c, a]
        bacodia1 = [-d, a]
        return aibacod0, aibacod1, bacodia0, bacodia1

    @staticmethod
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
            for j in range(1, C + 1):
                r[(i, j)] = vpool.id(f"{prefix}_r{i, j}")
        if m == 0 or C <= 0:
            raise ValueError(f"Yo What")
        clauses.extend(CNFBuilder.iff(var_list[0], r[1, 1]))
        for j in range(2, C + 1):
            clauses.append([-r[1, j]])
        for i in range(2, m + 1):
            clauses.extend(CNFBuilder.aiffboc(r[i, 1], var_list[i - 1], r[i - 1, 1]))
        for i in range(2, m + 1):
            for j in range(2, C + 1):
                clauses.extend(CNFBuilder.aiffbacod(r[i, j], var_list[i - 1], r[i - 1, j - 1], r[i - 1, j]))
        return clauses, vpool, r[m, C]


class CNFDebugger:
    @staticmethod
    def cnf_int_to_var_name(var, vpool):
        item = vpool.obj(abs(var))
        return item

    @staticmethod
    def print_cnf_vars(cnf):
        vars = []
        for inp in range(1, cnf.nv + 1):
            name = CNFDebugger.cnf_int_to_var_name(inp, cnf.vpool)
            if name is not None:
                vars.append(f"{inp, name}")
        print(vars)
        return vars

    @staticmethod
    def clause_translate(clauses, vpool):
        translated = []
        for clause in clauses:
            terms = []
            for term in clause:
                name = CNFDebugger.cnf_int_to_var_name(term, vpool)
                if name is None:
                    name = "?"
                if term < 0:
                    name = f"~{name}"
                terms.append(name)
            translated.append(terms)
        return translated

    @staticmethod
    def get_solver_model(solver, cnf):
        status = solver.get_status()
        if not status:
            print(f"Model has no solution")
            return
        model = solver.get_model()
        m = {}
        for var in model:
            name = CNFDebugger.cnf_int_to_var_name(var, cnf.vpool)
            if name is not None:
                m[name] = var > 0
        return m


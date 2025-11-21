from sympy import symbols, And, Not, Or, Implies, Equivalent
from sympy.logic.boolalg import truth_table

# Please replace "StudentMatriculationNumber" with your actual matric number here and in the filename
def A4_A0307665X(query):
    """
    Args:
        query: A sympy logical expression representing the query to be checked
               (e.g., A, Not(B), etc.)

    Returns:
        result: A string "True" if the query is a Knight;
                A string "False" if the query is a Knave;
                A string "Not Sure" if the type of the query cannot be determined.
    """

    
    A, B, C = symbols('A B C')  # A: Alex, B: Ben, C: Chloe (True = Knight)

   
    KB_Alex = Equivalent(A, Not(B))

    
    KB_Ben = Equivalent(B, Or(And(A, Not(B)), And(Not(A), B)))

    
    KB_Chloe = Equivalent(C, Or(A, C))

    KB = And(KB_Alex, KB_Ben, KB_Chloe)
    
    def all_models(symbols_list):
        n = len(symbols_list)
        for mask in range(1 << n):
            valuation = {}
            for i, sym in enumerate(symbols_list):
                valuation[sym] = bool((mask >> i) & 1)
            yield valuation

    
    def holds(expr, valuation):
        return bool(expr.subs(valuation))

   

    symbols_list = [A, B, C]
    entails_q = True
    entails_not_q = True
    has_model = False

    for valuation in all_models(symbols_list):
        if holds(KB, valuation):
            has_model = True
            q_val = holds(query, valuation)
            if q_val:
                entails_not_q = False
            else:
                entails_q = False

            if not entails_q and not entails_not_q:
                break  

    
    if not has_model:
        result = "Not Sure"
    elif entails_q and not entails_not_q:
        result = "True"
    elif entails_not_q and not entails_q:
        result = "False"
    else:
        result = "Not Sure"

    
    return result






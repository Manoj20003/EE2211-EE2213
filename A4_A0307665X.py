from sympy import symbols, And, Not, Or, Implies, Equivalent
from sympy.logic.boolalg import truth_table

# Please replace "StudentMatriculationNumber" with your actual matric number here and in the filename
def A4_StudentMatriculationNumber(query):
    """
    Args:
        query: A sympy logical expression representing the query to be checked
               (e.g., A, Not(B), etc.)

    Returns:
        result: A string "True" if the query is a Knight;
                A string "False" if the query is a Knave;
                A string "Not Sure" if the type of the query cannot be determined.
    """

    # Step 1: Define propositional symbols
    A, B, C = symbols('A B C')  # A: Alex, B: Ben, C: Chloe (True = Knight)

    # Step 2: Encode the Knowledge Base (KB)
    # Alex says: "Ben is a Knave" → A ⇔ ¬B
    KB_Alex = Equivalent(A, Not(B))

    # Ben says: "Alex and I are of different type" → B ⇔ (A ⊕ B)
    KB_Ben = Equivalent(B, Or(And(A, Not(B)), And(Not(A), B)))

    # Chloe says: "At least one of Alex and I is a Knight" → C ⇔ (A ∨ C)
    KB_Chloe = Equivalent(C, Or(A, C))

    KB = And(KB_Alex, KB_Ben, KB_Chloe)

    # Step 3: Helper – generate all possible truth assignments
    def all_models(symbols_list):
        n = len(symbols_list)
        for mask in range(1 << n):
            valuation = {}
            for i, sym in enumerate(symbols_list):
                valuation[sym] = bool((mask >> i) & 1)
            yield valuation

    # Step 4: Evaluate expression given a truth assignment
    def holds(expr, valuation):
        return bool(expr.subs(valuation))

    # Step 5: Model checking – check entailment
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
                break  # early exit

    # Step 6: Decide result
    if not has_model:
        result = "Not Sure"
    elif entails_q and not entails_not_q:
        result = "True"
    elif entails_not_q and not entails_q:
        result = "False"
    else:
        result = "Not Sure"

    # return in this order
    return result









from sympy import symbols, Not
A, B, C = symbols('A B C')

print(A4_StudentMatriculationNumber(A))     # "False"  → Alex is a Knave
print(A4_StudentMatriculationNumber(B))     # "True"   → Ben is a Knight
print(A4_StudentMatriculationNumber(C))     # "Not Sure" → Chloe’s type uncertain
print(A4_StudentMatriculationNumber(Not(A)))  # "True"
print(A4_StudentMatriculationNumber(Not(B)))  # "False"
print(A4_StudentMatriculationNumber(Not(C)))  # "Not Sure"

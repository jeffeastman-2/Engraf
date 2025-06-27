# atn/core.py
from engraf.lexer.token_stream import TokenStream

class ATNState:
    def __init__(self, name):
        self.name = name
        self.arcs = []

    def add_arc(self, test_fn, action_fn, next_state):
        self.arcs.append((test_fn, action_fn, next_state))

    def __repr__(self):
        return f"ATNState({self.name})"

def noop(ctx, tok):
    pass

# Walk the whole ATN to get all states
def get_all_states(start_state):
    visited = set()
    stack = [start_state]
    all_states = []

    while stack:
        state = stack.pop()
        if id(state) in visited:  # Use id to avoid hashing issues
            continue
        visited.add(id(state))
        all_states.append(state)

        for _, _, next_state in state.arcs:
            if isinstance(next_state, ATNState):
                stack.append(next_state)
            else:
                print(f"Warning: next_state {next_state} is not an ATNState")
    return all_states

def run_atn(start_state, end_state, ts: TokenStream, context=None):
    context = context or {}
    current = start_state

    while True:
        tok = ts.peek()
        matched = False

        for test, action, next_state in current.arcs:
            print(f"Testing in {current.name} → {next_state.name}")
            if test(tok):
                if tok is not None:
                    print(f"    Token '{tok.word}' matches in {current.name} → {next_state.name}")
                else: 
                    print(f"    Token is None, but matched in {current.name} → {next_state.name}")
                if action is None:
                    print("❌ ERROR: This arc has a None action!")
                action(context, tok)
                current = next_state
                 # ❗ Only advance if action is not a subnetwork runner
                if action != noop and not getattr(action, "_is_subnetwork", False):
                    ts.advance()
                matched = True
                break
            else:
                if tok is not None:
                    print(f"    Failed in {current.name} on '{tok.word}' → {next_state.name}")
                else:
                    print(f"    Failed in {current.name} on None → {next_state.name}")

        if not matched:
            if tok is not None:
                print(f"    ***No arc matched in {current.name} on token '{tok.word}'")
            else:           
                print(f"    ***No arc matched in {current.name} on None token")
            return None

        if current == end_state:
            print(f"Reached final state: {end_state.name}")
            return context


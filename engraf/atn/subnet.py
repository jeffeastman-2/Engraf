from engraf.lexer.token_stream import TokenStream
from engraf.atn.core import run_atn, get_all_states
from engraf.atn.np import build_np_atn
from engraf.atn.pp import build_pp_atn
from engraf.atn.vp import build_vp_atn
from engraf.utils.actions import diagnostic_stub

def make_run_np_into_ctx(ts: TokenStream):
    def run_np_into_ctx(ctx, tok):
        print(f"Running NP parser with token: {tok}")
        np_start, np_end = build_np_atn(ts)
        mark = ts.mark()
        np_result = run_atn(np_start, np_end, ts)
        if np_result is None:
            print("❌ NP parsing failed, rewinding token stream")
            ts.rewind(mark)
        else:
            ctx['object'] = np_result.get('noun')
            ctx['vector'] = np_result.get('vector')
            ctx['noun_phrase'] = np_result
    run_np_into_ctx._is_subnetwork = True
    return run_np_into_ctx

def make_run_pp_into_ctx(ts: TokenStream):
    def run_pp_into_ctx(ctx, tok):
        print(f"Running PP parser with token: {tok}")
        pp_start, pp_end = build_pp_atn(ts)
        mark = ts.mark()
        pp_result = run_atn(pp_start, pp_end, ts)
        if pp_result is None:
            print("❌ PP parsing failed, rewinding token stream")
            ts.rewind(mark)
        else:
            ctx.setdefault('modifiers', []).append(pp_result)
    run_pp_into_ctx._is_subnetwork = True
    return run_pp_into_ctx

def make_run_vp_into_ctx(ts: TokenStream):
    def run_vp_into_ctx(ctx, tok):
        print(f"Running VP parser with token: {tok}")
        vp_start, vp_end = build_vp_atn(ts)
        mark = ts.mark()
        vp_result = run_atn(vp_start, vp_end, ts)
        if vp_result is None:
            print("❌ VP parsing failed, rewinding token stream")
            ts.rewind(mark)
        else:
            # Merge VP result into parent context
            ctx.update(vp_result)
    run_vp_into_ctx._is_subnetwork = True  # Optional tagging
    return run_vp_into_ctx

def run_np(tokens):
    ts = TokenStream(tokens)
    ctx = {}

    np_start, np_end = build_np_atn(ts)
    print("Running NP ATN:")
    for state in get_all_states(np_start):
        for i, arc in enumerate(state.arcs):
            test, action, next_state = arc
            print(f"    {state.name} → {next_state.name} | Action: {action}")
            if action is diagnostic_stub and next_state.name == "NP-PP":
                patched = make_run_pp_into_ctx(ts)
                state.arcs[i] = (test, patched, next_state)
                print(f"        Patched action for NP-PP: {patched}")
    return run_atn(np_start, np_end, ts, ctx)

def run_pp(tokens):
    ts = TokenStream(tokens)
    ctx = {}

    pp_start, pp_end = build_pp_atn(ts)
    print("Running PP ATN:")
    for state in get_all_states(pp_start):
        for i, arc in enumerate(state.arcs):
            test, action, next_state = arc
            print(f"    {state.name} → {next_state.name} | Action: {action}")
            if action is diagnostic_stub and next_state.name == "PP-END":
                patched = make_run_np_into_ctx(ts)
                state.arcs[i] = (test, patched, next_state)
                print(f"        Patched action for PP-END: {patched}")
    return run_atn(pp_start, pp_end, ts, ctx)

def run_vp(tokens):
    ts = TokenStream(tokens)
    ctx = {}

    vp_start, vp_end = build_vp_atn(ts)
    print("Running VP ATN:")
    for state in get_all_states(vp_start):
        for test, action, next_state in state.arcs:
            print(f"    {state.name} → {next_state.name} | Action: {action}")

    return run_atn(vp_start, vp_end, ts, ctx)
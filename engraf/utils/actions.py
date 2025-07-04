from engraf.atn.core import run_atn
from engraf.lexer.token_stream import TokenStream

def make_run_np_into_ctx(ts):
    from engraf.atn.np import build_np_atn

    def run_np_into_ctx(ctx, _):
        np_ctx = {}
        np_start, np_end = build_np_atn(ts)
        result = run_atn(np_start, np_end, ts, np_ctx)
        ctx['object'] = result.get('noun')
        ctx['noun_phrase'] = result 
        ctx['vector'] = result.get('vector', None)
        ctx['modifiers'] = result.get('modifiers', None)
    run_np_into_ctx._is_subnetwork = True
    return run_np_into_ctx

def make_run_pp_into_ctx(ts):
    from engraf.atn.pp import build_pp_atn

    def run_pp_into_ctx(ctx, tok):
        ctx.setdefault('modifiers', []).append({'prep': tok})
        modifier_ctx = ctx['modifiers'][-1]
        pp_start, pp_end = build_pp_atn(ts)
        result = run_atn(pp_start, pp_end, ts, modifier_ctx)
        if result is not None:
            modifier_ctx['object'] = result.get('object')
            if "vector" not in ctx:
                ctx["vector"] = result.get("vector")
    run_pp_into_ctx._is_subnetwork = True
    return run_pp_into_ctx


def make_run_vp_into_ctx(ts, output_key="vector"):
    from engraf.atn.vp import build_vp_atn
    
    def run_vp_into_ctx(ctx, _):
        vp_ctx = {}
        vp_start, vp_end = build_vp_atn(ts)
        result = run_atn(vp_start, vp_end, ts, vp_ctx)
        if result:
            ctx[output_key] = result.get("vector")
            ctx["object"] = result.get("object")  # optional: noun or pronoun
    return run_vp_into_ctx

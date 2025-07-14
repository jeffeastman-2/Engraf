from engraf.atn.core import ATNState
from engraf.lexer.token_stream import TokenStream
from engraf.utils.predicates import is_verb, is_none, is_np_head, is_conjunction_no_consume, is_preposition, any_of, is_tobe, is_adjective, is_number, is_conjunction, is_adjective_conjunction, is_predicate_conjunction
from engraf.atn.core import noop
from engraf.utils.actions import make_run_np_into_atn, make_run_pp_into_atn, make_run_coordinated_np_into_atn
from engraf.pos.verb_phrase import VerbPhrase
from engraf.utils.actions import make_run_np_into_atn, make_run_pp_into_atn, apply_from_subnet, apply_from_subnet_multi, make_run_coordinated_np_into_atn


# --- Build the Verb Phrase ATN ---
def build_vp_atn(vp: VerbPhrase, ts: TokenStream):
    start = ATNState("VP-START")
    after_verb = ATNState("VP-NP")
    after_np = ATNState("VP-AFTER-NP")
    after_adj = ATNState("VP-AFTER-ADJ")
    adj_conj = ATNState("VP-ADJ-CONJ")  # New state for adjective conjunctions
    amount = ATNState("VP-AMOUNT")
    after_amount = ATNState("VP-AFTER-AMOUNT")
    pp = ATNState("VP-PP")
    end = ATNState("VP-END")

    # VERB
    start.add_arc(is_verb, lambda _, tok: vp.apply_verb(tok), after_verb)
    # NP (subnetwork) - supports coordinated noun phrases for objects like "a cube and a sphere"
    after_verb.add_arc(is_np_head, make_run_coordinated_np_into_atn(ts, fieldname="noun_phrase"), after_np)    
    # Allow final transition if stream is exhausted
    after_verb.add_arc(is_none, noop, end)

    # After NP, can have amount (like "45 degrees"), PP, adjective complement, or end
    after_np.add_arc(is_number, make_run_coordinated_np_into_atn(ts, fieldname="amount"), after_amount)
    after_np.add_arc(is_preposition, make_run_pp_into_atn(ts), pp)
    after_np.add_arc(is_adjective, lambda _, tok: vp.apply_adjective(tok), after_adj)
    after_np.add_arc(is_none, apply_from_subnet("noun_phrase", vp.apply_np), end)
    # Allow VP to end when conjunction is encountered (don't consume it)
    after_np.add_arc(is_conjunction_no_consume, apply_from_subnet("noun_phrase", vp.apply_np), end)
    # Allow VP to end when other verbs/tobe are encountered
    after_np.add_arc(any_of(is_verb, is_tobe), apply_from_subnet("noun_phrase", vp.apply_np), end)
    
    # After amount, can have PP or end
    after_amount.add_arc(is_preposition, make_run_pp_into_atn(ts), pp)
    after_amount.add_arc(is_none, apply_from_subnet_multi("noun_phrase", vp.apply_np, "amount", vp.apply_amount), end)
    after_amount.add_arc(is_conjunction_no_consume, apply_from_subnet_multi("noun_phrase", vp.apply_np, "amount", vp.apply_amount), end)
    after_amount.add_arc(any_of(is_verb, is_tobe), apply_from_subnet_multi("noun_phrase", vp.apply_np, "amount", vp.apply_amount), end)
    
    # After adjective complement, can have conjunction for coordinated adjectives or end
    after_adj.add_arc(is_adjective_conjunction(ts), lambda _, tok: None, adj_conj)  # Handle "and" between adjectives
    after_adj.add_arc(is_predicate_conjunction(ts), apply_from_subnet("noun_phrase", vp.apply_np), end)  # Handle predicate conjunction
    after_adj.add_arc(is_none, apply_from_subnet("noun_phrase", vp.apply_np), end)
    after_adj.add_arc(any_of(is_verb, is_tobe), apply_from_subnet("noun_phrase", vp.apply_np), end)
    
    # After adjective conjunction, expect another adjective
    adj_conj.add_arc(is_adjective, lambda _, tok: vp.apply_adjective(tok), after_adj)
    
    # After PP, apply both NP and PP then end
    pp.add_arc(is_none, apply_from_subnet_multi("noun_phrase", vp.apply_np, "preposition", vp.apply_pp), end)
    pp.add_arc(is_conjunction_no_consume, apply_from_subnet_multi("noun_phrase", vp.apply_np, "preposition", vp.apply_pp), end)
    pp.add_arc(any_of(is_verb, is_tobe), apply_from_subnet_multi("noun_phrase", vp.apply_np, "preposition", vp.apply_pp), end)
    
    # Allow final transition if stream is exhausted
    end.add_arc(is_none, noop, end)
    
    return start, end



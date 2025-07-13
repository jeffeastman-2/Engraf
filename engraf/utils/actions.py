from engraf.atn.core import run_atn
from engraf.lexer.token_stream import TokenStream

def make_run_np_into_atn(ts, fieldname):
    from engraf.atn.np import build_np_atn
    from engraf.pos.noun_phrase import NounPhrase

    def run_np_into_atn(parent_atn, _):
        saved_pos = ts.position
        np_obj = NounPhrase()
        np_start, np_end = build_np_atn(np_obj, ts)
        result = run_atn(np_start, np_end, ts, np_obj)

        if result is not None:
            print(f"✅ Set {fieldname} = {result}")  # ADD THIS LINE
            setattr(parent_atn, fieldname, result)
            return parent_atn
        else:
            ts.position = saved_pos
            return None

    run_np_into_atn._is_subnetwork = True
    return run_np_into_atn

def make_run_pp_into_atn(ts):
    from engraf.atn.pp import build_pp_atn
    from engraf.pos.prepositional_phrase import PrepositionalPhrase

    def run_pp_into_atn(parent_atn, _):
        saved_pos = ts.position
        pp_obj = PrepositionalPhrase()
        pp_start, pp_end = build_pp_atn(pp_obj, ts)
        result = run_atn(pp_start, pp_end, ts, pp_obj)

        if result is not None:
            parent_atn.preps.append(pp_obj)
            return parent_atn
        else:
            ts.position = saved_pos  # Roll back if it failed
            return None

    run_pp_into_atn._is_subnetwork = True
    return run_pp_into_atn


def make_run_vp_into_atn(ts, fieldname):
    from engraf.atn.vp import build_vp_atn
    from engraf.pos.verb_phrase import VerbPhrase

    def run_vp_into_atn(parent_atn, _):
        saved_pos = ts.position
        vp_obj = VerbPhrase()
        vp_start, vp_end = build_vp_atn(vp_obj, ts)
        result = run_atn(vp_start, vp_end, ts, vp_obj)

        if result is not None:
            setattr(parent_atn, fieldname, vp_obj)
            return parent_atn  # ✅ Return the parent sentence!
        else:
            ts.position = saved_pos
            return None
    
    run_vp_into_atn._is_subnetwork = True
    return run_vp_into_atn

def make_run_vp_into_conjunction(ts):
    """Create a VP action that adds the VP to the existing conjunction."""
    from engraf.atn.vp import build_vp_atn
    from engraf.pos.verb_phrase import VerbPhrase
    from engraf.pos.conjunction_phrase import ConjunctionPhrase

    def run_vp_into_conjunction(parent_atn, _):
        saved_pos = ts.position
        vp_obj = VerbPhrase()
        vp_start, vp_end = build_vp_atn(vp_obj, ts)
        result = run_atn(vp_start, vp_end, ts, vp_obj)

        if result is not None:
            # Add the VP to the existing conjunction
            if isinstance(parent_atn.predicate, ConjunctionPhrase):
                tail = parent_atn.predicate.get_last()
                if tail.right is None:
                    tail.right = vp_obj
                else:
                    # This shouldn't happen in simple cases, but handle it
                    tail.right = ConjunctionPhrase(tail.right.conjunction, left=tail.right, right=vp_obj)
            else:
                # This shouldn't happen if conjunction was applied first
                print("⚠️  Warning: Expected ConjunctionPhrase but got", type(parent_atn.predicate))
            return parent_atn
        else:
            ts.position = saved_pos
            return None
    
    run_vp_into_conjunction._is_subnetwork = True
    return run_vp_into_conjunction


def make_run_sentence_into_atn(ts):
    from engraf.atn.sentence import build_sentence_atn, Sentence

    def run_sentence_into_atn(atn, _):
        saved_pos = ts.position
        sent_start, sent_end = build_sentence_atn(ts)

        sentence_obj = Sentence()
        result = run_atn(sent_start, sent_end, ts, sentence_obj)

        if result is not None:
            atn.sentence = sentence_obj
            atn.vector = sentence_obj.vector
        else:
            ts.position = saved_pos

    run_sentence_into_atn._is_subnetwork = True
    return run_sentence_into_atn

def apply_from_subnet(fieldname, apply_func):
    def wrapper(_, atn):
        val = getattr(atn, fieldname, None)
        print(f"📍 apply_from_subnet: {fieldname} = {val}")
        if val is not None:
            apply_func(val)
        else:
            print(f"⚠️  Warning: {fieldname} is None — skipping apply_func")
    return wrapper

def apply_from_subnet_multi(*field_func_pairs):
    """Apply multiple fields from subnet with their respective functions.
    Usage: apply_from_subnet_multi("field1", func1, "field2", func2, ...)
    """
    def wrapper(_, atn):
        for i in range(0, len(field_func_pairs), 2):
            fieldname = field_func_pairs[i]
            apply_func = field_func_pairs[i + 1]
            val = getattr(atn, fieldname, None)
            print(f"📍 apply_from_subnet_multi: {fieldname} = {val}")
            if val is not None:
                apply_func(val)
            else:
                print(f"⚠️  Warning: {fieldname} is None — skipping apply_func")
    return wrapper



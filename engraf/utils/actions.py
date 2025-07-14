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

def make_run_np_into_conjunction(ts):
    """Create an NP action that adds the NP to the existing conjunction."""
    from engraf.atn.np import build_np_atn
    from engraf.pos.noun_phrase import NounPhrase
    from engraf.pos.conjunction_phrase import ConjunctionPhrase

    def run_np_into_conjunction(parent_atn, _):
        saved_pos = ts.position
        np_obj = NounPhrase()
        np_start, np_end = build_np_atn(np_obj, ts)
        result = run_atn(np_start, np_end, ts, np_obj)

        if result is not None:
            # Add the NP to the existing conjunction
            if isinstance(parent_atn, ConjunctionPhrase):
                tail = parent_atn.get_last()
                if tail.right is None:
                    tail.right = np_obj
                else:
                    # This shouldn't happen in simple cases, but handle it
                    tail.right = ConjunctionPhrase(tail.right.conjunction, left=tail.right, right=np_obj)
            else:
                # This shouldn't happen if conjunction was applied first
                print("⚠️  Warning: Expected ConjunctionPhrase but got", type(parent_atn))
            return parent_atn
        else:
            ts.position = saved_pos
            return None
    
    run_np_into_conjunction._is_subnetwork = True
    return run_np_into_conjunction

def make_run_coordinated_np_into_atn(ts, fieldname):
    """Create an NP action that can handle coordinated noun phrases."""
    from engraf.atn.np import build_np_atn
    from engraf.pos.noun_phrase import NounPhrase
    from engraf.pos.conjunction_phrase import ConjunctionPhrase

    def run_coordinated_np_into_atn(parent_atn, _):
        saved_pos = ts.position
        first_np = NounPhrase()
        np_start, np_end = build_np_atn(first_np, ts)
        result = run_atn(np_start, np_end, ts, first_np)

        if result is not None:
            # Check if there's a conjunction after the first NP
            if ts.peek() and ts.peek().isa("conj"):
                # Save position before consuming conjunction
                conj_pos = ts.position
                conj_tok = ts.next()
                
                # Try to parse the second NP
                second_np = NounPhrase()
                np_start, np_end = build_np_atn(second_np, ts)
                second_result = run_atn(np_start, np_end, ts, second_np)
                
                if second_result is not None:
                    # Successfully parsed coordinated NPs
                    coord_np = ConjunctionPhrase(conj_tok, left=first_np, right=second_np)
                    
                    # Check for more conjunctions (recursive)
                    while ts.peek() and ts.peek().isa("conj"):
                        conj_tok = ts.next()
                        next_np = NounPhrase()
                        np_start, np_end = build_np_atn(next_np, ts)
                        next_result = run_atn(np_start, np_end, ts, next_np)
                        
                        if next_result is not None:
                            # Chain the conjunction
                            coord_np.right = ConjunctionPhrase(conj_tok, left=coord_np.right, right=next_np)
                        else:
                            break
                    
                    print(f"✅ Set {fieldname} = {coord_np} (coordinated)")
                    setattr(parent_atn, fieldname, coord_np)
                    return parent_atn
                else:
                    # Failed to parse second NP - this might be a predicate conjunction
                    # Rollback to before the conjunction and return just the first NP
                    ts.position = conj_pos
                    print(f"✅ Set {fieldname} = {result} (simple, conjunction not consumed)")
                    setattr(parent_atn, fieldname, result)
                    return parent_atn
            else:
                # No conjunction, just a simple NP
                print(f"✅ Set {fieldname} = {result} (simple)")
                setattr(parent_atn, fieldname, result)
                return parent_atn
        else:
            ts.position = saved_pos
            return None

    run_coordinated_np_into_atn._is_subnetwork = True
    return run_coordinated_np_into_atn



from engraf.atn.core import run_atn
from engraf.lexer.token_stream import TokenStream

def make_run_np_into_atn(ts):
    from engraf.atn.np import build_np_atn
    from engraf.pos.noun_phrase import NounPhrase

    def run_np_into_atn(atn, _):
        saved_pos = ts.position
        np_obj = NounPhrase()
        np_start, np_end = build_np_atn(np_obj, ts)
        result = run_atn(np_start, np_end, ts, np_obj)

        if result is not None:
            atn.noun_phrase = np_obj
            atn.vector = np_obj.vector
            atn.object = np_obj.noun
            atn.modifiers = getattr(np_obj, "modifiers", None)
        else:
            ts.position = saved_pos

    run_np_into_atn._is_subnetwork = True
    return run_np_into_atn

def make_run_pp_into_atn(ts):
    from engraf.atn.pp import build_pp_atn
    from engraf.pos.prepositional_phrase import PrepositionalPhrase

    def run_pp_into_atn(atn, _):
        saved_pos = ts.position
        pp_obj = PrepositionalPhrase()
        pp_start, pp_end = build_pp_atn(pp_obj, ts)
        result = run_atn(pp_start, pp_end, ts, pp_obj)

        if result is not None:
            atn.noun_phrase = pp_obj  # 👈 used by apply_pp() caller
            atn.vector = pp_obj.vector
        else:
            ts.position = saved_pos

    run_pp_into_atn._is_subnetwork = True
    return run_pp_into_atn


def make_run_vp_into_atn(ts):
    from engraf.atn.vp import build_vp_atn
    from engraf.pos.verb_phrase import VerbPhrase

    def run_vp_into_atn(atn, _):
        saved_pos = ts.position
        vp_obj = VerbPhrase()
        vp_start, vp_end = build_vp_atn(vp_obj)
        result = run_atn(vp_start, vp_end, ts, vp_obj)

        if result is not None:
            atn.verb_phrase = vp_obj
            atn.vector += vp_obj.vector
            atn.action = getattr(vp_obj, "action", None)
        else:
            ts.position = saved_pos

    run_vp_into_atn._is_subnetwork = True
    return run_vp_into_atn

def make_run_sentence_into_atn(ts):
    from engraf.atn.sentence import build_sentence_atn
    from engraf.pos.sentence import Sentence

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

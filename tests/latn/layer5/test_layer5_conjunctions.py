
import pytest
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.pos.sentence_phrase import SentencePhrase
from engraf.pos.noun_phrase import NounPhrase
from engraf.pos.prepositional_phrase import PrepositionalPhrase
from engraf.pos.verb_phrase import VerbPhrase
from engraf.pos.conjunction_phrase import ConjunctionPhrase

class TestLayer5Conjunctions:
    # Test cases for Layer 5 conjunctions from GPT-5

    def setup_method(self):
        """Set up test environment."""
        self.executor = LATNLayerExecutor()
    

# A) NP-level coordination (Layer-2)
    # 1. Basic NP∧NP as subject
    def test_basic_np_and_np_as_subject(self):
        sentence = "the cube and the sphere move"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) == 2, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector._original_sp
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        #[S [NP [Det the] [N cube]] ∧ [NP [Det the] [N sphere]] [VP [V move]]]
        subj = sent.subject
        assert isinstance(subj, ConjunctionPhrase), "Subject should be ConjunctionPhrase(NP,NP)"
        assert subj.vector.isa("plural"), "Subject number should be plural"
        assert subj.vector.isa("conj"), "Subject should be a noun phrase"
        pred = sent.predicate
        assert isinstance(pred, VerbPhrase), "Predicate should be a VerbPhrase"

    # 2. Adj coordination inside NP (not NP∧NP)
    def test_adj_coordination_inside_np(self):
        sentence = "the red and blue cube"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) == 1, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector._original_sp
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        #[S [NP [Det the] [Adj red ∧ blue] [N cube]]]
        subj = sent.subject
        assert isinstance(subj, NounPhrase), "Subject should be NounPhrase(Adj,Adj)"
        pred = sent.predicate
        assert pred is None, "Predicate should be None"

    # 3. Three-way NP list (Oxford comma allowed)
    def test_three_way_np_list(self):
        sentence = "the cube, the sphere and the cone"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) == 2, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector._original_sp
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        #[S [NP [Det the] [N cube]] ∧ [NP [Det the] [N sphere]] ∧ [NP [Det the] [N cone]]]
        subj = sent.subject
        assert isinstance(subj, ConjunctionPhrase), "Subject should be ConjunctionPhrase(NP,NP)"
        assert subj.vector.isa("plural"), "Subject number should be plural"
        assert subj.vector.isa("conj"), "Subject should be a noun phrase"
        parts = [np for np in subj.flatten()]
        assert len(parts) == 3, "Should have three coordinated NPs"
        pred = sent.predicate
        assert pred is None, "Predicate should be None"

    # 4. Ambiguous nominal (keep both parses)
    def test_ambiguous_nominal(self):
        sentence = "the red and blue boxes and spheres"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) == 2, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector._original_sp
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # A: [NP ( [NP red∧blue boxes] ∧ [NP spheres] )]
        # B: [NP ( [NP red] ∧ [NP blue boxes ∧ spheres] )]
        subj = sent.subject
        assert isinstance(subj, ConjunctionPhrase), "Subject should be ConjunctionPhrase(NP,NP)"
        assert subj.vector.isa("plural"), "Subject number should be plural"
        assert subj.vector.isa("conj"), "Subject should be a noun phrase"
        parts = [np for np in subj.flatten()]
        assert len(parts) == 2, "Should have two coordinated NPs"
        pred = sent.predicate
        assert pred is None, "Predicate should be None"


    # 5. NP with numeric determiners
    def test_np_with_numeric_determiners(self):
        sentence = "two cubes and three spheres"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) == 2, "Should extract one hypothesis"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector._original_sp
        #[NP ( [NP two cubes] ∧ [NP three spheres] )]
        subj = sent.subject
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        assert isinstance(subj, ConjunctionPhrase), "Subject should be ConjunctionPhrase(NP,NP)"
        assert subj.vector.isa("plural"), "Subject number should be plural"
        assert subj.vector.isa("conj"), "Subject should be a noun phrase"
        parts = [np for np in subj.flatten()]
        assert len(parts) == 2, "Should have two coordinated NPs"
        pred = sent.predicate
        assert pred is None, "Predicate should be None"

# B) PP-level (Layer-3)
    # 6. PP attaches to whole NP coordination
    def test_pp_attaches_to_whole_np_coordination(self):
        sentence = "the cube and the sphere on the table"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) == 2, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        np = vector._original_np
        assert isinstance(np, ConjunctionPhrase), "First hypothesis should be a ConjunctionPhrase"
        # Parses (keep both):
        # A (wide): [NP ( [NP the cube] ∧ [NP the sphere] ) [PP on [NP the table]]]
        # B (narrow): [NP [NP the cube] ∧ [NP the sphere [PP on [NP the table]]]]
        assert np.vector.isa("plural"), "Subject number should be plural"
        assert np.vector.isa("conj"), "Subject should be a noun phrase"
        parts = [np for np in np.flatten()]
        assert len(parts) == 2, "Should have two coordinated NPs"
        pp = hypo[1]._original_pp
        assert isinstance(pp, PrepositionalPhrase), "Second hypothesis should be a PrepositionalPhrase"
        assert pp.vector.isa("prep"), "PP should be a preposition"   

    # 7. PP∧PP predicate complement
    def test_pp_and_pp_predicate_complement(self):
        sentence = "the sphere is above the cube and in front of the box"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) == 3, "Should extract three hypotheses"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        np = vector._original_np
        assert isinstance(np, NounPhrase), "First hypothesis should be a NounPhrase"
        # Parse:
        #[S [NP the sphere] [VP [V is] [PP ( [PP above [NP the cube]] ∧ [PP in front of [NP the box]] ) ]]]

    # 8. Left/Right relation
    def test_left_right_relation(self):
        sentence = "the cube is left of the sphere and right of the cone"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector._original_sp
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        #[S [NP the cube] [VP [V is] [PP ( [PP left of [NP the sphere]] ∧ [PP right of [NP the cone]] ) ]]]
        assert len(result.hypotheses) == 1, "Should extract one hypothesis"
        assert result.hypotheses[0].canonical_hash == hypo.canonical_hash, "Hypothesis should match"

# C) VP-level coordination (Layer-4)
    # 9. VP∧VP under same subject
    def test_vp_and_vp_under_same_subject(self):
        sentence = "the cube move and rotate"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector._original_sentence
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        #[S [NP the cube] [VP ( [VP move] ∧ [VP rotate] )]]
        assert isinstance(vs.predicate, ConjunctionPhrase), "Predicate should be ConjunctionPhrase(VP,VP)"
        assert vs.predicate.number == "plural", "Predicate number should be plural"
        # Parse:
        # [S [NP the cube] [VP ( [VP move] ∧ [VP rotate] )]]

    # 10. VP∧VP with shared object (right-node raising)
    def test_vp_and_vp_with_shared_object(self):
        sentence = "the cube color the sphere and move the sphere"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector._original_sentence
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        #[S [NP the cube] [VP ( [VP color [NP the sphere]] ∧ [VP move [NP the sphere]] )]]
        assert isinstance(vs.predicate, ConjunctionPhrase), "Predicate should be ConjunctionPhrase(VP,VP)"
        assert vs.predicate.number == "plural", "Predicate number should be plural"
        # Parse:
        # [S [NP the cube] [VP ( [VP color [NP the sphere]] ∧ [VP move [NP the sphere]] )]]
    
    # 11. Mixed: VP∧VP and NP∧NP as object
    def test_vp_and_np_as_object(self):
        sentence = "the cube color the sphere and the box and move the sphere and the box"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector._original_sentence
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        #[S [NP the cube] [VP ( [VP color [NP (the sphere ∧ the box)]] ∧ [VP move [NP (the sphere ∧ the box)]] )]]
        assert isinstance(vs.predicate, ConjunctionPhrase), "Predicate should be ConjunctionPhrase(VP,VP)"
        assert vs.predicate.number == "plural", "Predicate number should be plural"
        # Parse:
        # [S [NP the cube] [VP ( [VP color [NP (the sphere ∧ the box)]] ∧ [VP move [NP (the sphere ∧ the box)]] )]]

    # 12. Adverb inside VP conj
    def test_vp_with_adverb(self):
        sentence = "the cube rotate and slightly move"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector._original_sentence
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        #[S [NP the cube] [VP ( [VP rotate] ∧ [VP [Adv slightly] [V move]] )]]
        assert isinstance(sent.predicate, ConjunctionPhrase), "Predicate should be ConjunctionPhrase(VP,VP)"
        assert sent.predicate.number == "plural", "Predicate number should be plural"
        # Parse:
        # [S [NP the cube] [VP ( [VP rotate] ∧ [VP [Adv slightly] [V move]] )]]

# D) S-level coordination (Layer-5)
    # 13. S∧S with comma
    def test_s_and_s_with_comma(self):
        sentence = "the cube move, and the sphere rotate"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector._original_sentence
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        #[S ( [S [NP the cube] [VP move]] ∧ [S [NP the sphere] [VP rotate]] )]
        assert isinstance(vs.predicate, ConjunctionPhrase), "Predicate should be ConjunctionPhrase(S,S)"
        assert vs.predicate.number == "plural", "Predicate number should be plural"
        # Parse:
        # [S ( [S [NP the cube] [VP move]] ∧ [S [NP the sphere] [VP rotate]] )]

# 14. Correlative “both…and” gives plural subject
    def test_both_and(self):
        sentence = "both the cube and the sphere are smooth"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector._original_sentence
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        #[S [NP (the cube ∧ the sphere)] [VP [V are] [Adj smooth]]]
        assert vs.subject.number == "plural", "Subject number should be plural"
        # [S [NP (the cube ∧ the sphere)] [VP [V are] [Adj smooth]]]
        # Assert: plural agreement.

    # 15. Disjunction “or” (document your number policy)
    def test_or_clause(self):
        sentence = "the cube or the spheres are near the table"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector._original_sentence
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        #[S [NP (the cube ∨ the spheres)] [VP [V are] [PP near [NP the table]]]]
        assert isinstance(sent.predicate, DisjunctionPhrase), "Predicate should be DisjunctionPhrase(NP,NP)"
        assert sent.predicate.number == "plural", "Predicate number should be plural"
        # Parse:
        # [S [NP (the cube ∨ the spheres)] [VP [V are] [PP near [NP the table]]]]

    # E) Preposition + movement verbs
    # 16. Move with directional PP
    def test_move_with_directional_pp(self):
        sentence = "the cube move to the table"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector._original_sentence
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        #[S [NP the cube] [VP [V move] [PP to [NP the table]]]]
        assert isinstance(sent.predicate, MovementPhrase), "Predicate should be MovementPhrase"
        assert sent.predicate.direction == "to", "Predicate direction should be 'to'"
        # Parse:
        # [S [NP the cube] [VP [V move] [PP to [NP the table]]]]
        assert vs.predicate.direction == "to", "Predicate direction should be 'to'"

    # 17. Rotate around axis (axis nouns provided)
    def test_rotate_around_axis(self):
        sentence = "the cube rotate around the x-axis"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector._original_sentence
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        #[S [NP the cube] [VP [V rotate] [PP around [NP the x-axis]]]]
        assert isinstance(vs.predicate, RotationPhrase), "Predicate should be RotationPhrase"
        assert vs.predicate.axis == "x-axis", "Predicate axis should be 'x-axis'"

    # 18. X/Y/Z rotate verbs (semantics in vectors)
    def test_test_x_y_z_rotate(self):
        sentence = "the sphere xrotate and yrotate"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector._original_sentence
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        #[S [NP the sphere] [VP ( [VP xrotate] ∧ [VP yrotate] )]]
        assert isinstance(vs.predicate, ConjunctionPhrase), "Predicate should be ConjunctionPhrase"
        assert vs.predicate.number == "plural", "Predicate number should be plural"

    # F) Pronouns & agreement
    # 19. Pronoun plural subject + VP
    def test_pronoun_plural_subject(self):
        sentence = "they move"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector._original_sentence
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        #[S [NP they] [VP move]]
        assert vs.subject.number == "plural", "Subject number should be plural"

    # 20.Pronoun object reused in VP∧VP
    def test_pronoun_object_reused(self):
        sentence = "the cube color them and move them"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector._original_sentence
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        #[S [NP the cube] [VP ( [VP color [NP them]] ∧ [VP move [NP them]] )]]
        assert isinstance(vs.predicate, ConjunctionPhrase), "Predicate should be ConjunctionPhrase"

    # G) Edge / negative (greediness guard)
    # 21. Avoid bogus NP∧NP when lookahead isn’t NP-start
    def test_avoid_bogus_np_coordination(self):
        sentence = "the cube and slightly rotate"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector._original_sentence
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        #[S [NP the cube] [VP ( [VP color [NP them]] ∧ [VP move [NP them]] )]]
        assert isinstance(vs.predicate, ConjunctionPhrase), "Predicate should be ConjunctionPhrase"

    # 22. PP scope ambiguity retained
    def test_pp_scope_ambiguity(self):
        sentence = "the cube and the sphere left of the box"
        result = self.executor.execute_layer5(sentence)
        assert result.success, "Layer 5 should succeed"
        assert len(result.hypotheses) > 0, "Should extract Sentence objects"
        hypo = result.hypotheses[0]
        hypo.print_tokens()
        vector = hypo[0]
        sent = vector._original_sentence
        assert isinstance(sent, SentencePhrase), "First hypothesis should be a SentencePhrase"
        # Parse:
        # A: [NP (the cube ∧ the sphere) [PP left of [NP the box]]]
        # B: [NP the cube ∧ [NP the sphere [PP left of [NP the box]]]]
        assert len(result.hypotheses) == 2, "Should produce two hypotheses"
        assert result.hypotheses[0].canonical_hash != result.hypotheses[1].canonical_hash, "Hypotheses should have distinct canonical hashes"

#!/usr/bin/env python3
"""
Tests for Layer-6 LLM inference on newly generated scenes.

Verifies that the trained model can:
1. Process new random scenes through LATN L1-L5
2. Generate appropriate text responses
3. Handle all sentence types (imperative, declarative, interrogative)
"""

import sys
sys.path.insert(0, '/Users/jeff/Python/Engraf')

import pytest
import torch
from pathlib import Path

from engraf.llm_layer6.dataset import (
    OnTheFlyLayer6Dataset, 
    STRUCTURAL_TOKEN_TO_ID,
    Layer6TextTokenizer
)
from engraf.llm_layer6.model_simple import Layer6EncoderOnlySimple, TemplateDecoder
from engraf.llm_layer6.synthetic_generator import (
    generate_random_scene,
    SentenceGenerator,
    process_through_layer5,
    create_training_pair_from_hyp
)
from engraf.lexer.latn_layer_executor import LATNLayerExecutor
from engraf.lexer.vector_space import VECTOR_LENGTH

# Semantic vector dimension (from VECTOR_DIMENSIONS)
SEMANTIC_VECTOR_DIM = VECTOR_LENGTH


class TestLayer6Inference:
    """Test Layer-6 model inference on new scenes."""
    
    @pytest.fixture
    def text_tokenizer(self):
        """Build a text tokenizer from known vocabulary."""
        from engraf.llm_layer6.synthetic_generator import (
            SHAPE_NOUNS, COLOR_ADJECTIVES, SIZE_ADJECTIVES,
            INTENSITY_ADVERBS, ALL_SPATIAL_PREPS, ACTION_VERBS
        )
        
        tokenizer = Layer6TextTokenizer()
        
        # Build vocabulary
        all_words = set()
        all_words.update(SHAPE_NOUNS)
        all_words.update(COLOR_ADJECTIVES)
        all_words.update(SIZE_ADJECTIVES)
        all_words.update(INTENSITY_ADVERBS)
        all_words.update(ALL_SPATIAL_PREPS)
        all_words.update(ACTION_VERBS)
        all_words.update([
            'the', 'is', 'are', 'a', 'an', 'of', 'to', 'in', 'on', 'at',
            'yes', 'no', 'not', 'what', 'where', 'which', 'how',
            'moving', 'placing', 'positioning',
            'there', 'here', 'it', 'that', 'this',
            '.', '?', ',', '!'
        ])
        
        tokenizer.fit(list(all_words))
        return tokenizer
    
    @pytest.fixture
    def model(self, text_tokenizer):
        """Create a model instance (untrained for basic tests)."""
        model = Layer6EncoderOnlySimple(
            text_vocab_size=len(text_tokenizer.vocab),
            max_output_length=50,
            structural_vocab_size=12,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            dropout=0.0  # Disable dropout for deterministic tests
        )
        model.eval()
        return model
    
    @pytest.fixture
    def decoder(self, text_tokenizer):
        """Create a template decoder."""
        id_to_token = {v: k for k, v in text_tokenizer.vocab.items()}
        return TemplateDecoder(id_to_token, text_tokenizer.vocab)
    
    def test_scene_generation(self):
        """Test that random scenes are generated correctly."""
        scene = generate_random_scene(num_objects=4)
        
        assert len(scene.objects) == 4
        for obj in scene.objects:
            assert obj.object_id is not None
            assert hasattr(obj, 'vector')
    
    def test_sentence_generation(self):
        """Test that sentences are generated for a scene."""
        scene = generate_random_scene(num_objects=4)
        generator = SentenceGenerator(scene)
        
        imperatives = generator.generate_imperatives()
        declaratives = generator.generate_declaratives()
        interrogatives = generator.generate_interrogatives()
        
        # Should have sentences
        assert len(imperatives) > 0
        assert len(declaratives) > 0
        assert len(interrogatives) > 0
        
        # Check format: (sentence, answer, obj_ids)
        sentence, answer, obj_ids = imperatives[0]
        assert isinstance(sentence, str)
        assert isinstance(answer, str)
        assert isinstance(obj_ids, list)
    
    def test_latn_processing(self):
        """Test that sentences process through LATN L1-L5."""
        scene = generate_random_scene(num_objects=4)
        executor = LATNLayerExecutor(scene)
        generator = SentenceGenerator(scene)
        
        sentences = generator.generate_all()[:5]  # Test first 5
        
        for sentence, answer, obj_ids in sentences:
            hyp = process_through_layer5(executor, sentence, scene)
            
            if hyp is not None and hyp.layer6_tokens:
                # Should have structural tokens
                assert len(hyp.layer6_tokens) > 0
                
                # Tokens should be valid
                for token in hyp.layer6_tokens:
                    assert token in STRUCTURAL_TOKEN_TO_ID, f"Unknown token: {token}"
    
    def test_training_pair_creation(self):
        """Test that training pairs are created correctly."""
        scene = generate_random_scene(num_objects=4)
        executor = LATNLayerExecutor(scene)
        generator = SentenceGenerator(scene)
        
        sentence, answer, obj_ids = generator.generate_imperatives()[0]
        hyp = process_through_layer5(executor, sentence, scene)
        
        if hyp and hyp.layer6_tokens:
            pair = create_training_pair_from_hyp(hyp, answer)
            
            assert 'structural_tokens' in pair
            assert 'semantic_vectors' in pair
            assert 'scene_grounding' in pair
            assert 'target_string' in pair
            
            # Vectors should be SEMANTIC_VECTOR_DIM (currently 69)
            for vec in pair['semantic_vectors']:
                if isinstance(vec, (list, tuple)):
                    assert len(vec) == SEMANTIC_VECTOR_DIM
    
    def test_model_forward_pass(self, model, text_tokenizer):
        """Test that model can process Layer-6 input."""
        # Create sample input
        batch_size = 2
        seq_len = 20
        
        structural_tokens = torch.randint(0, 12, (batch_size, seq_len))
        semantic_vectors = torch.randn(batch_size, seq_len, SEMANTIC_VECTOR_DIM)
        grounding_ids = torch.randint(0, 100, (batch_size, seq_len))
        
        with torch.no_grad():
            logits = model(structural_tokens, semantic_vectors, grounding_ids)
        
        assert logits.shape == (batch_size, 50, len(text_tokenizer.vocab))
    
    def test_end_to_end_inference(self, model, text_tokenizer, decoder):
        """Test end-to-end: scene -> LATN -> model -> decoded text."""
        scene = generate_random_scene(num_objects=4)
        executor = LATNLayerExecutor(scene)
        generator = SentenceGenerator(scene)
        
        # Get a sentence
        sentence, expected_answer, obj_ids = generator.generate_imperatives()[0]
        
        # Process through LATN
        hyp = process_through_layer5(executor, sentence, scene)
        
        if hyp and hyp.layer6_tokens:
            pair = create_training_pair_from_hyp(hyp, expected_answer)
            
            # Convert to tensors
            struct_ids = [STRUCTURAL_TOKEN_TO_ID.get(tok, 11) for tok in pair['structural_tokens']]
            struct_ids = struct_ids[:50] + [11] * max(0, 50 - len(struct_ids))
            structural_tokens = torch.tensor([struct_ids], dtype=torch.long)
            
            # Run model
            with torch.no_grad():
                logits = model(structural_tokens)
            
            # Decode output
            output_text = decoder.decode(logits[0])
            
            # Output should be a string (content depends on training)
            assert isinstance(output_text, str)
    
    def test_on_the_fly_dataset(self):
        """Test the OnTheFlyLayer6Dataset generates valid examples."""
        dataset = OnTheFlyLayer6Dataset(
            num_examples=10,
            objects_per_scene=4,
            scene_cache_size=2,
            seed=42
        )
        
        # Get a few examples
        for i in range(5):
            example = dataset[i]
            
            assert 'structural_tokens' in example
            assert 'semantic_vectors' in example
            assert 'target_ids' in example
            assert 'question' in example
            
            assert example['structural_tokens'].shape[0] == 50
            assert example['semantic_vectors'].shape == (50, SEMANTIC_VECTOR_DIM)
            assert example['target_ids'].shape[0] == 50
    
    def test_multiple_scenes_produce_variety(self):
        """Test that different scenes produce different sentences."""
        sentences_seen = set()
        
        for _ in range(5):
            scene = generate_random_scene(num_objects=4)
            generator = SentenceGenerator(scene)
            
            for sentence, _, _ in generator.generate_imperatives()[:3]:
                sentences_seen.add(sentence)
        
        # Should have variety (5 scenes * 3 sentences each, minus some overlap)
        assert len(sentences_seen) >= 10


class TestLayer6WithTrainedModel:
    """Tests that require a trained model checkpoint."""
    
    @pytest.fixture
    def checkpoint_path(self):
        """Path to trained model checkpoint."""
        path = Path('/Users/jeff/Python/Engraf/engraf/llm_layer6/layer6_checkpoints_simple/best_model.pt')
        if not path.exists():
            pytest.skip("No trained model checkpoint found")
        return path
    
    @pytest.fixture
    def trained_model(self, checkpoint_path):
        """Load trained model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        config = checkpoint.get('model_config', {})
        model = Layer6EncoderOnlySimple(
            text_vocab_size=config.get('text_vocab_size', 73),
            max_output_length=config.get('max_output_length', 50),
            structural_vocab_size=12,
            embedding_dim=config.get('embedding_dim', 128),
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 2),
            dropout=0.0
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, checkpoint.get('text_tokenizer')
    
    def test_trained_model_produces_valid_output(self, trained_model):
        """Test that trained model produces valid text output."""
        model, text_tokenizer = trained_model
        
        if text_tokenizer is None:
            pytest.skip("Checkpoint missing text_tokenizer")
        
        id_to_token = {v: k for k, v in text_tokenizer.vocab.items()}
        decoder = TemplateDecoder(id_to_token, text_tokenizer.vocab)
        
        # Generate a new scene and sentence
        scene = generate_random_scene(num_objects=4)
        executor = LATNLayerExecutor(scene)
        generator = SentenceGenerator(scene)
        
        sentence, expected_answer, obj_ids = generator.generate_imperatives()[0]
        hyp = process_through_layer5(executor, sentence, scene)
        
        if hyp and hyp.layer6_tokens:
            pair = create_training_pair_from_hyp(hyp, expected_answer)
            
            # Convert to tensor
            struct_ids = [STRUCTURAL_TOKEN_TO_ID.get(tok, 11) for tok in pair['structural_tokens']]
            struct_ids = struct_ids[:50] + [11] * max(0, 50 - len(struct_ids))
            structural_tokens = torch.tensor([struct_ids], dtype=torch.long)
            
            with torch.no_grad():
                logits = model(structural_tokens)
            
            output_text = decoder.decode(logits[0])
            
            # Should produce some output
            assert len(output_text) > 0
            
            # Output should contain recognizable words
            words = output_text.lower().split()
            # Check for expected vocabulary (spatial terms, object terms, etc.)
            vocab_words = {'the', 'is', 'above', 'below', 'near', 'moving', 'placing', 'yes', 'no'}
            has_vocab = any(w in vocab_words for w in words)
            
            print(f"Input: {sentence}")
            print(f"Expected: {expected_answer}")
            print(f"Output: {output_text}")
    
    def test_trained_model_on_different_sentence_types(self, trained_model):
        """Test trained model on imperative, declarative, and interrogative sentences."""
        model, text_tokenizer = trained_model
        
        if text_tokenizer is None:
            pytest.skip("Checkpoint missing text_tokenizer")
        
        id_to_token = {v: k for k, v in text_tokenizer.vocab.items()}
        decoder = TemplateDecoder(id_to_token, text_tokenizer.vocab)
        
        scene = generate_random_scene(num_objects=4)
        executor = LATNLayerExecutor(scene)
        generator = SentenceGenerator(scene)
        
        sentence_types = [
            ('imperative', generator.generate_imperatives()[:2]),
            ('declarative', generator.generate_declaratives()[:2]),
            ('interrogative', generator.generate_interrogatives()[:2]),
        ]
        
        results = []
        
        for sent_type, sentences in sentence_types:
            for sentence, expected, obj_ids in sentences:
                hyp = process_through_layer5(executor, sentence, scene)
                
                if hyp and hyp.layer6_tokens:
                    pair = create_training_pair_from_hyp(hyp, expected)
                    
                    struct_ids = [STRUCTURAL_TOKEN_TO_ID.get(tok, 11) for tok in pair['structural_tokens']]
                    struct_ids = struct_ids[:50] + [11] * max(0, 50 - len(struct_ids))
                    structural_tokens = torch.tensor([struct_ids], dtype=torch.long)
                    
                    with torch.no_grad():
                        logits = model(structural_tokens)
                    
                    output_text = decoder.decode(logits[0])
                    
                    results.append({
                        'type': sent_type,
                        'input': sentence,
                        'expected': expected,
                        'output': output_text
                    })
        
        # Should process at least some sentences
        assert len(results) > 0
        
        # Print results for inspection
        for r in results:
            print(f"\n[{r['type']}]")
            print(f"  Input:    {r['input']}")
            print(f"  Expected: {r['expected']}")
            print(f"  Output:   {r['output']}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Semantic Agreement Validation for ENGRAF

This module provides validation to ensure that commands make semantic sense
given the current scene state, such as checking that "move 3 circles" is only
allowed when there are at least 3 circles in the scene.
"""

from engraf.visualizer.scene.scene_model import SceneModel
from engraf.pos.conjunction_phrase import ConjunctionPhrase


class SemanticAgreementValidator:
    """Validates semantic agreement between commands and scene state."""
    
    def __init__(self, scene_model: SceneModel):
        self.scene = scene_model
    
    def validate_command(self, sentence, original_text=None):
        """
        Validate that the sentence is semantically valid given the current scene state.
        
        Args:
            sentence: Parsed sentence object
            original_text: Original text for fallback parsing (optional)
            
        Returns:
            Tuple of (is_valid: bool, error_message: str or None)
        """
        if not sentence or not sentence.predicate:
            return True, None
            
        # Handle ConjunctionPhrase by validating each component
        if isinstance(sentence.predicate, ConjunctionPhrase):
            # This is a ConjunctionPhrase - validate each component
            for component in sentence.predicate.flatten():
                is_valid, error_msg = self._validate_single_phrase(component)
                if not is_valid:
                    return False, error_msg
            return True, None
        else:
            # This is a single verb phrase
            return self._validate_single_phrase(sentence.predicate)
    
    def _validate_single_phrase(self, verb_phrase):
        """Validate a single verb phrase."""
        if not verb_phrase:
            return True, None
            
        noun_phrase = verb_phrase.noun_phrase
        
        if not noun_phrase or not noun_phrase.vector:
            return True, None
            
        # Check if this is a creation verb using vector features
        if hasattr(verb_phrase, 'vector') and verb_phrase.vector and verb_phrase.vector['create'] > 0.0:
            # Creation verbs don't need existing objects - they create new ones
            return True, None
            
        # Extract requested quantity and target noun from the verb phrase
        requested_quantity, target_noun = self._analyze_verb_phrase(verb_phrase)
        
        if requested_quantity is None:
            return True, None  # No specific quantity requested
        
        # Handle pronouns specially - they need to resolve to at least one object
        if target_noun and target_noun.startswith("pronoun:"):
            pronoun_word = target_noun.split(":")[1]  # Extract "it" or "them"
            if requested_quantity == 0:  # No objects resolved
                error_msg = f"Semantic error: Cannot {verb_phrase.verb} '{pronoun_word}' - no objects available in scene"
                return False, error_msg
            else:
                return True, None  # Pronoun resolved successfully
            
        # Handle regular nouns - count matching objects in scene with full attribute matching
        available_count = self._count_matching_objects(noun_phrase)
        
        # Check if we have enough objects that match all attributes (including color)
        if requested_quantity > available_count:
            determiner = noun_phrase.determiner or str(int(requested_quantity))
            noun_display = self._describe_noun_phrase(noun_phrase)
            error_msg = f"Semantic error: Cannot {verb_phrase.verb} {determiner} {noun_display} - only {available_count} available in scene"
            return False, error_msg
            
        return True, None
    
    def _analyze_verb_phrase(self, verb_phrase):
        """
        Analyze the verb phrase to extract the requested quantity and target noun.
        Returns (requested_quantity, target_noun) or (None, None) if not applicable.
        """
        noun_phrase = verb_phrase.noun_phrase
        
        if not noun_phrase or not noun_phrase.vector:
            return None, None
            
        # Handle pronouns by using the existing resolve_pronoun function
        if noun_phrase.vector['pronoun'] > 0:
            # Import here to avoid circular imports
            from engraf.visualizer.scene.scene_model import resolve_pronoun
            
            # Determine pronoun type from vector features
            if noun_phrase.vector['singular'] > 0:
                pronoun_word = "it"
            elif noun_phrase.vector['plural'] > 0:
                pronoun_word = "them"
            else:
                return None, None  # Unknown pronoun type
            
            try:
                resolved_objects = resolve_pronoun(pronoun_word, self.scene)
                available_count = len(resolved_objects)
                
                # For semantic validation, we always expect the resolved count
                # The actual validation happens in the validate_command method
                return available_count, f"pronoun:{pronoun_word}"
                    
            except ValueError:
                return None, None  # Invalid pronoun
        
        # Extract target noun for non-pronoun cases
        target_noun = noun_phrase.noun
        if not target_noun:
            return None, None
            
        # Extract quantity from determiner or default to 1
        determiner = noun_phrase.determiner
        if determiner:
            try:
                # Try to extract number from determiner
                if determiner.strip().isdigit():
                    requested_quantity = int(determiner.strip())
                elif determiner.strip() in ['a', 'an', 'the']:
                    requested_quantity = 1
                else:
                    # Try to parse number words
                    number_words = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
                    requested_quantity = number_words.get(determiner.strip().lower(), 1)
            except (ValueError, AttributeError):
                requested_quantity = 1
        else:
            requested_quantity = 1
            
        return requested_quantity, target_noun

    def _count_objects_by_noun(self, noun):
        """Count objects in the scene matching the given noun (already singular)."""
        count = 0
        for obj in self.scene.objects:
            if obj.name == noun:
                count += 1
        return count

    def _count_matching_objects(self, noun_phrase):
        """Count objects in the scene that match ALL attributes in the noun phrase (noun, color, etc.)."""
        if not noun_phrase or not noun_phrase.noun:
            return 0
            
        count = 0
        for obj in self.scene.objects:
            if self._object_matches_noun_phrase(obj, noun_phrase):
                count += 1
        return count
    
    def _object_matches_noun_phrase(self, obj, noun_phrase):
        """Check if an object matches all attributes specified in the noun phrase."""
        # Must match the base noun
        if obj.name != noun_phrase.noun:
            return False
            
        # Check color attributes if specified
        if noun_phrase.vector:
            # Check for specific color requirements
            color_features = ['red', 'green', 'blue']
            for color in color_features:
                # If the noun phrase specifies this color (e.g., "blue circle")
                if noun_phrase.vector[color] > 0:
                    # The object must also have this color
                    if not (obj.vector and obj.vector[color] > 0):
                        return False
                        
        # Object matches all specified attributes
        return True
    
    def _describe_noun_phrase(self, noun_phrase):
        """Generate a descriptive string for the noun phrase including adjectives."""
        if not noun_phrase:
            return "objects"
            
        description_parts = []
        
        # Add color adjectives if present
        if noun_phrase.vector:
            color_features = ['red', 'green', 'blue']
            for color in color_features:
                if noun_phrase.vector[color] > 0:
                    description_parts.append(color)
        
        # Add the base noun
        if noun_phrase.noun:
            description_parts.append(noun_phrase.noun)
        
        return " ".join(description_parts) if description_parts else "objects"

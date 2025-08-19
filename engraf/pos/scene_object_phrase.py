from engraf.pos.noun_phrase import NounPhrase
from engraf.lexer.vector_space import VectorSpace
from engraf.utils.debug import debug_print


class SceneObjectPhrase(NounPhrase):
    """
    SceneObjectPhrase - A subclass of NounPhrase that represents a resolved scene object.
    
    This class represents NPs that have been successfully grounded to scene objects
    through LATN Layer 2 semantic grounding. It adds the SO part of speech marker
    and manages the resolved object relationship.
    """
    
    def __init__(self, noun=None, source_np=None):
        """Initialize a SceneObjectPhrase.
        
        Args:
            noun: The noun token (optional)
            source_np: A NounPhrase to base this SceneObjectPhrase on (optional)
        """
        if source_np:
            # Copy all attributes from the source NounPhrase
            super().__init__(noun)
            self.vector = source_np.vector.copy() if source_np.vector else VectorSpace()
            self.noun = source_np.noun
            self.pronoun = source_np.pronoun
            self.determiner = source_np.determiner
            self.preps = source_np.preps.copy()
            self.scale_factor = source_np.scale_factor
            self.proper_noun = source_np.proper_noun
            self.consumed_tokens = source_np.consumed_tokens.copy()
            # Move resolved_object from the source NP if it exists
            self.resolved_object = getattr(source_np, 'resolved_object', None)
        else:
            super().__init__(noun)
            self.resolved_object = None
        
        # Add the SO part of speech marker to the vector
        self.vector["SO"] = 1.0
    
    def resolve_to_scene_object(self, scene_object):
        """Resolve this SO to a specific SceneObject (LATN Layer 2 semantic grounding).
        
        Args:
            scene_object: The SceneObject this SO refers to
        """
        self.resolved_object = scene_object
        
    def is_resolved(self):
        """Check if this SO has been resolved to a scene object."""
        return self.resolved_object is not None
        
    def get_resolved_object(self):
        """Get the resolved SceneObject, or None if not resolved."""
        return self.resolved_object
    
    def __repr__(self):
        consumed_words = [tok.word for tok in self.consumed_tokens]
        parts = [f"noun={self.noun}", f"determiner={self.determiner}", f"vector={self.vector}", f"PPs={self.preps}", f"consumed={consumed_words}"]
        if self.resolved_object:
            parts.append(f"resolved_to={self.resolved_object.object_id}")
        return f"SO({', '.join(parts)})"
    
    @classmethod
    def from_noun_phrase(cls, np):
        """Create a SceneObjectPhrase from an existing NounPhrase.
        
        Args:
            np: The NounPhrase to convert
            
        Returns:
            A new SceneObjectPhrase with copied attributes
        """
        return cls(source_np=np)

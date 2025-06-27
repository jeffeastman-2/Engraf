class SceneObject:
    def __init__(self, name, vector, modifiers=None):
        self.name = name                  # e.g., 'cube'
        self.vector = vector              # VectorSpace instance
        self.modifiers = modifiers or []  # nested SceneObjects from PPs

    def __repr__(self):
        return f"<{self.name} {self.vector} modifiers={self.modifiers}>"

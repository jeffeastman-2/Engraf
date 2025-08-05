"""
Temporal Scene Management for ENGRAF

This module provides temporal navigation functionality, allowing users to
go back and forward in time through scene states. Each command execution
creates a new scene snapshot, enabling undo/redo functionality.
"""

from typing import List, Optional
from engraf.visualizer.scene.scene_model import SceneModel


class TemporalScenes:
    """Manages temporal navigation through scene states."""
    
    def __init__(self, initial_scene: Optional[SceneModel] = None):
        """
        Initialize temporal scenes with an optional initial scene.
        
        Args:
            initial_scene: Starting scene state. If None, creates empty scene.
        """
        self.scenes: List[SceneModel] = [initial_scene or SceneModel()]
        self.current_index: int = 0
    
    def get_current_scene(self) -> SceneModel:
        """Get the current scene state."""
        return self.scenes[self.current_index]
    
    def add_scene_snapshot(self, scene: SceneModel) -> None:
        """
        Add a new scene state after executing a command.
        
        This truncates any future history if we're not at the end,
        then adds the new scene state.
        
        Args:
            scene: The new scene state to add
        """
        # Truncate future history if we're not at the end
        self.scenes = self.scenes[:self.current_index + 1]
        
        # Add new snapshot (deep copy to ensure independence)
        self.scenes.append(scene.copy())
        self.current_index += 1
    
    def go_back(self) -> bool:
        """
        Go back in time to the previous scene state.
        
        Returns:
            True if successful, False if already at the beginning
        """
        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False
    
    def go_forward(self) -> bool:
        """
        Go forward in time to the next scene state.
        
        Returns:
            True if successful, False if already at the end
        """
        if self.current_index < len(self.scenes) - 1:
            self.current_index += 1
            return True
        return False
    
    def can_go_back(self) -> bool:
        """Check if we can go back in time."""
        return self.current_index > 0
    
    def can_go_forward(self) -> bool:
        """Check if we can go forward in time."""
        return self.current_index < len(self.scenes) - 1
    
    def get_scene_count(self) -> int:
        """Get the total number of scene states."""
        return len(self.scenes)
    
    def get_current_index(self) -> int:
        """Get the current scene index (0-based)."""
        return self.current_index
    
    def get_scene_at_index(self, index: int) -> Optional[SceneModel]:
        """
        Get the scene at a specific index.
        
        Args:
            index: The scene index (0-based)
            
        Returns:
            The scene at the index, or None if index is invalid
        """
        if 0 <= index < len(self.scenes):
            return self.scenes[index]
        return None
    
    def clear_history(self, keep_current: bool = True) -> None:
        """
        Clear the temporal history.
        
        Args:
            keep_current: If True, keeps only the current scene. If False, resets to empty scene.
        """
        if keep_current and self.scenes:
            current_scene = self.get_current_scene()
            self.scenes = [current_scene.copy()]
        else:
            self.scenes = [SceneModel()]
        self.current_index = 0
    
    def __len__(self) -> int:
        """Return the number of scene states."""
        return len(self.scenes)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"TemporalScenes(scenes={len(self.scenes)}, current={self.current_index})"

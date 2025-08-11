# Engraf

🎨 **A sophisticated natural language processing framework that transforms spoken commands into interactive 3D scenes.**

Engraf combines advanced linguistic parsing with real-time 3D visualization, allowing users to create and manipulate virtual objects through natural language commands like "draw a large red cube" or "move it to the left."

## ✨ Key Features

### 🧠 Advanced NLP Engine
- **ATN (Augmented Transition Network)** parsing with hierarchical grammar support
- **Semantic vector spaces** with 40+ dimensional embeddings for rich meaning representation
- **Comparative/superlative adjective handling** (bigger, biggest, redder, reddest)
- **Pronoun resolution** and contextual reference tracking
- **Multi-object coordination** with conjunction support

### 🎬 3D Scene Generation
- **Real-time VPython rendering** with interactive 3D visualizations
- **Object creation** via natural language (cubes, spheres, cylinders, cones, etc.)
- **Dynamic transformations** (move, rotate, scale) with coordinate precision
- **Color and material properties** from descriptive adjectives
- **Spatial relationship understanding** (above, below, near, at coordinates)

### 🔧 Robust Architecture
- **Modular interpreter system** with specialized handlers for different command types
- **Scene state management** with object tracking and history
- **Error handling** with graceful fallbacks and informative feedback
- **Comprehensive test suite** with 220+ unit tests covering edge cases

## 🚀 Quick Demo

```python
from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.vpython_renderer import VPythonRenderer

# Create interpreter with 3D visualization
renderer = VPythonRenderer(title="My 3D Scene")
interpreter = SentenceInterpreter(renderer=renderer)

# Natural language commands create 3D objects
interpreter.interpret("draw a large red cube")
interpreter.interpret("draw a blue sphere above it")
interpreter.interpret("move the cube to [3, 0, 0]")
interpreter.interpret("rotate the sphere by 45 degrees")
```

## 📁 Project Structure

```
engraf/
├── atn/                    # Augmented Transition Network parsing
│   ├── core.py            # Core ATN state machine
│   ├── sentence.py        # Sentence-level grammar rules
│   ├── np.py              # Noun phrase parsing
│   ├── vp.py              # Verb phrase parsing
│   └── pp.py              # Prepositional phrase parsing
├── interpreter/            # High-level command interpretation
│   ├── sentence_interpreter.py  # Main interpreter orchestration
│   └── handlers/          # Specialized command handlers
│       ├── object_creator.py    # Object creation logic
│       ├── object_modifier.py   # Transformation handling
│       ├── object_resolver.py   # Reference resolution
│       └── scene_manager.py     # Scene state management
├── lexer/                 # Tokenization and vocabulary
│   ├── token_stream.py    # Token processing pipeline
│   ├── vocabulary.py      # Semantic vector space definitions
│   └── vector_space.py    # Multi-dimensional embeddings
├── pos/                   # Part-of-speech phrase structures
│   ├── noun_phrase.py     # NP data structures
│   ├── verb_phrase.py     # VP data structures
│   └── prepositional_phrase.py  # PP data structures
├── visualizer/            # 3D rendering and scene management
│   ├── renderers/         # Rendering backends
│   │   ├── vpython_renderer.py  # VPython 3D visualization
│   │   └── mock_renderer.py     # Testing renderer
│   ├── scene/             # Scene state and objects
│   │   ├── scene_model.py       # Scene graph management
│   │   └── scene_object.py      # 3D object representations
│   └── transforms/        # Geometric transformations
│       └── transform_matrix.py       # Matrix operations
└── utils/                 # Utility functions
    ├── actions.py         # Action definitions
    ├── noun_inflector.py  # Plural/singular handling
    └── predicates.py      # Logical predicates

tests/                     # Comprehensive test suite (220+ tests)
├── interpreter/           # Interpreter integration tests
├── visualizer/           # Rendering and scene tests
└── [linguistic_tests]/   # Grammar and parsing tests
```

## 🎯 What You Can Say

Engraf understands a rich variety of natural language constructs:

### Object Creation
```
"draw a red cube"
"create a large blue sphere"
"make a tiny green cylinder and a huge yellow cone"
"build a very tall white pyramid at [0, 5, 0]"
```

### Object Modification
```
"move the cube to [3, 0, 0]"
"rotate it by 45 degrees"
"scale the sphere by [2, 1, 2]"
"color the pyramid orange"
```

### Spatial Relationships
```
"place a sphere above the cube"
"draw a cylinder near the red object"
"move it under the large sphere"
"position the cone at the origin"
```

### Complex Commands
```
"draw a bigger red cube than the blue one"
"create the smallest possible green sphere"
"make three large pyramids in a row"
"rotate all the objects by 90 degrees"
```

## 🧬 Technical Highlights

### Semantic Vector Spaces
Each word is represented in a 40+ dimensional semantic space capturing:
- **Grammatical properties**: part-of-speech, number, case
- **Spatial semantics**: location, scale, rotation coordinates  
- **Visual properties**: color (RGB), texture, transparency
- **Verb semantics**: create, transform, style, organize, edit, select
- **Relational meaning**: spatial relationships, comparisons, possession

### Advanced Parsing
- **Hierarchical ATN grammar** with recursive phrase structures
- **Contextual disambiguation** using semantic vectors
- **Pronoun resolution** with scene state tracking
- **Comparative forms** with automatic base form lookup and intensity scaling
- **Coordinate parsing** for precise spatial positioning

### 3D Rendering Pipeline
- **VPython integration** for real-time browser-based visualization
- **Object lifecycle management** with creation, update, and deletion
- **Transform composition** with proper matrix operations
- **Material property mapping** from linguistic descriptions to visual attributes

## 🛠️ Getting Started

### Prerequisites

- **Python 3.12+** (tested with 3.12.4)
- **VPython** for 3D visualization
- **NumPy** for vector operations
- **pytest** for running tests

### Installation

```bash
git clone https://github.com/jeffeastman-2/Engraf.git
cd Engraf

# Install dependencies
pip install vpython numpy pytest

# Verify installation with test suite
pytest tests/
```

### Quick Start

```python
# Run the interactive demo
python demo_sentence_interpreter.py

# Or use programmatically
from engraf.interpreter.sentence_interpreter import SentenceInterpreter
from engraf.visualizer.renderers.vpython_renderer import VPythonRenderer

# Create 3D scene interpreter
renderer = VPythonRenderer(width=1000, height=700, title="My Scene")
interpreter = SentenceInterpreter(renderer=renderer)

# Start creating with natural language!
result = interpreter.interpret("draw a red cube")
print(f"Created: {result['objects_created']}")

result = interpreter.interpret("move it to [2, 2, 0]")
print(f"Modified: {result['objects_modified']}")
```

## 🧪 Development & Testing

### Running Tests
```bash
# Run all tests (220+ test cases)
pytest tests/ -v

# Run specific test categories
pytest tests/interpreter/ -v          # Interpreter tests
pytest tests/visualizer/ -v           # Rendering tests
pytest tests/test_sentence.py -v      # Grammar tests
```

### Key Test Coverage
- ✅ **Grammar parsing**: All major linguistic constructs
- ✅ **Semantic processing**: Vector space operations and word lookup
- ✅ **Scene generation**: Object creation, modification, and spatial relationships
- ✅ **Rendering pipeline**: VPython integration and visual updates
- ✅ **Error handling**: Graceful failure modes and recovery
- ✅ **Edge cases**: Pronoun resolution, coordinate parsing, comparative adjectives

### Architecture Patterns
- **Handler-based interpretation** for clean separation of concerns
- **Immutable scene objects** with copy-on-write semantics
- **Functional vector operations** with NumPy backends
- **State machine parsing** with ATN transitions
- **Dependency injection** for renderer flexibility

## 🎓 Research Applications

Engraf demonstrates several advanced NLP and computer graphics concepts:
- **Semantic embedding spaces** for linguistic meaning representation
- **Grammar-driven parsing** with recursive phrase structures  
- **Multimodal interaction** between language and 3D visualization
- **Contextual reference resolution** in dynamic environments
- **Transform composition** for complex 3D manipulations

## 🤝 Contributing

Contributions welcome! Areas of active development:
- **Extended vocabulary** for more object types and properties
- **Animation support** for temporal commands ("rotate slowly")
- **Physics integration** for realistic object interactions
- **Natural language queries** ("what objects are red?")
- **Export capabilities** for generated scenes

## 📊 Project Stats

- **220+ unit tests** with comprehensive coverage
- **40+ semantic dimensions** in vector space
- **12+ grammatical constructs** supported
- **6+ 3D object types** with extensible architecture
- **4+ coordinate systems** (Cartesian, relative, spatial relationships)

## 📜 License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

**Built with ❤️ for natural language understanding and 3D visualization**

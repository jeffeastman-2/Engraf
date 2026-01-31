# ***Layered Augmented Transition Networks (LATN): A Modular Approach to Language Parsing and Grounding***

## Abstract

This paper introduces a novel layered architecture for natural language
understanding based on Augmented Transition Networks (ATNs), termed
**Layered ATN (LATN)**. Inspired by the layered processing seen in
transformer models, LATN decomposes the parsing process into multiple
passes. Each pass incrementally builds syntactic and semantic
representations of an input sentence. Grounding occurs incrementally
within each layer, leveraging a persistent scene model to disambiguate
references and guide interpretation. This modular design provides a
transparent and interpretable framework for processing language in
simple visual domains and offers a foundation for more complex grounded
language models.

## Introduction

Engraf (**Eastman, J.F. 1975**), the system on which LATN is based, was
an early natural language understanding framework that combined
vector-semantic word representations (**Rumelhart, Hinton & Williams
1986**) (a precursor to modern Transformer embeddings) with an
executable 3D scene model **(Eastman, 2025)**. In Engraf, nouns denoted
displayable objects in a persistent computer graphics world, adjectives
modified a noun's features by adding their embedding vectors to the
noun's embedding vector (analogous to Value-vector addition in
contemporary Attention (**Vaswani, A., et al. 2017**) mechanisms), and
verbs acted as mathematical operators that transformed those objects.
Although limited by the hardware of its era, Engraf demonstrated a fully
integrated pipeline for parsing, grounding, and executing the semantics
of imperative sentences in a 3D environment. This approach parallels the
grounded reasoning explored in contemporary systems such as
SHRDLU **(Winograd, 1972)**, though Engraf incorporated an explicitly
vector-semantic model.

LATN extends this lineage by introducing a modern, multi-layered ATN
(**Woods, W. A. 1970**) architecture that separates lexical, syntactic,
and semantic processing while retaining Engraf's core principle: meaning
is realized through grounded operations on a scene model rather than
through text alone.

## Natural language processing systems traditionally face challenges when attempting to parse and interpret complex sentences in a single pass. Classical ATNs---including the original Engraf---operate within a single, densely interconnected control flow that forces early commitments and handles ambiguity poorly. LATN addresses this limitation by distributing the parsing process across multiple layers that separately handle lexical analysis, noun phrase extraction, prepositional phrase resolution, verb phrase construction, and anaphora resolution. Each layer builds on the output of the previous one and passes a modified, enriched representation forward. Ambiguity is managed by allowing each layer to maintain and propagate multiple hypotheses, a strategy analogous to beam search **(Graves, 2012)** in modern sequence models, ensuring that plausible interpretations remain available for later semantic or spatial disambiguation. This layered design echoes the multiple levels of abstraction found in Transformer architectures, where earlier layers capture local syntactic structure and later layers support deeper semantic reasoning. Each LATN layer processes a shared but evolving representation of the input---effectively forming a Vector-Symbolic Architecture **(Plate, 1995)** that blends symbolic ATN structure, Engraf-style semantic vectors, and grounded Scene Object references.

## Layer‑1: Lexical Analysis

Layer‑1 performs lexical processing on the input sentence. It handles
one-, two-, and three-word tokens by searching for each token in the
vocabulary. Each token is represented as a 70+-dimensional semantic
vector encoding features such as object type, color, size, position,
and action semantics. If not found directly, it attempts to identify
plural noun forms and verb/adjective inflections, adding the appropriate
embedding-vector features to the base word's vector when applicable.

This is the opposite of transformer tokenization, where input words are
fragmented into learned, sub-word tokens and later recombined through
learned attention. If LATN's multi-word tokens or inflectional variants
yield ambiguous embeddings, Layer‑1 generates multiple hypotheses.

**Output:** A list of lexical hypotheses containing token vectors and
token types, analogous to a beam of alternatives passed to higher
layers.

## Layer‑2: Noun Phrases

Layer‑2 invokes a shared ATN parsing engine to identify all possible
Noun Phrase (NP) structures within each lexical hypothesis. The shared
parsing algorithm is parameterized by a layer-specific ATN and the desired
non-terminal to be produced. Ambiguous parses create additional
hypotheses.

Once NPs are identified, Layer‑2 Grounding compares each NP's embedding
vector with objects in the *Scene Model*, a persistent memory structure
containing all previously created Scene Objects. NPs that match grounded
Scene Objects are linked to those objects, making the referent available
to all subsequent layers.

Grounding occurs here because noun phrases introduce the primary
discourse entities. Once grounded, these entities become part of the
persistent world state, enabling the system to resolve pronouns and
definite descriptions without revisiting earlier sentences.

**Output:** Grounded NP structures, each referencing specific Scene
Objects where they exist.

## Layer‑3: Prepositional Phrases

Layer‑3 identifies all Prepositional Phrases (PPs) in each hypothesis
using the shared ATN engine. After PP identification, Layer‑3 Grounding
attempts to attach each PP to a previous grounded NP, generating several
hypotheses. Then, using spatial validation logic tuned to the
preposition's embedding features, invalid hypotheses are removed.

LATN evaluates PP attachments using geometric and spatial consistency
checks derived from the grounded Scene Objects. This approach resolves
many classic PP ambiguities without probabilistic scoring.

**Output:** NP structures augmented with grounded PPs, with unresolvable
cases spawning additional hypotheses.

## Layer‑4: Verb Phrases

Layer‑4 identifies Verb Phrases (VPs) through ATN parsing, then uses
Layer‑4 Grounding to validate the VP's verb semantics, arguments, and
associated PPs or scalar/vector modifiers.

Verbs are treated as *vector‑space operators*: - additive
transformations (e.g., MOVE, TRANSLATE) - multiplicative or
compositional transformations (e.g., ROTATE, SCALE)

These operators implement Engraf's original design, where verbs define
executable mathematical transformations over grounded Scene Objects.

Hypotheses failing semantic validation at this level are also
eliminated.

**Output:** A transformation plan specifying updates to Scene Object
states.

## Layer‑5: Sentence Phrases

Layer‑5 identifies Imperative, Declarative, and Interrogative Sentence
Phrase (SP) structures. Once an SP‑node is formed, Layer‑5 Grounding
validates its semantics and eliminates hypotheses that produce
incompatible or multiple SP‑structures.

A grounded SP‑node yields a fully specified operation or sequence of
operations executable on the scene model.

**Output:** A complete grounded interpretation of the sentence, ready
for execution.

**EXAMPLE:**

Consider the input sentence: *"Place the sphere below the cube behind
the cylinder."*\
The Scene Model initially contains four grounded objects: a table, a red
cube, a blue sphere, and a green cylinder.

1.  **Layer-1** identifies the tokens and their corresponding embedding
    vectors.

2.  **Layer-2** detects three Noun Phrases (*the sphere, the cube, the
    cylinder*) and grounds them to Scene
    Objects \<blue_sphere_1\>, \<red_cube_1\>, and \<green_cylinder_1\>.

3.  **Layer-3** identifies two Prepositional Phrases (*below the
    cube*, *behind the cylinder*) and generates several hypotheses for
    PP attachment. Spatial validation eliminates all but one attachment
    configuration.

4.  **Layer-4** forms the Verb Phrase *(place \<blue_sphere_1\>
    ...)* and verifies the geometric feasibility of applying the spatial
    transformation.

5.  **Layer-5** produces a single grounded Sentence Phrase representing
    the executable operation:\
     **move(blue_sphere_1, (below red_cube_1 (behind
    green_cylinder_1)))**

This compact example illustrates how LATN uses layered parsing,
grounding, and spatial validation to converge on the single semantically
correct interpretation.
## Layer-6: LLM Integration

Layer-6 extends LATN beyond deterministic parsing into neural language
model territory. While Layers 1–5 produce grounded, executable
interpretations through symbolic processing, Layer-6 bridges these
structured representations with learned neural models.

The Layer-6 architecture uses an encoder-only transformer that processes
LATN's structural tokens ([NP, ]NP, [VP, ]VP, etc.) alongside the
70+-dimensional semantic vectors produced by earlier layers. This hybrid
approach preserves LATN's interpretability while enabling:

-   **Template-based generation** — producing natural language responses
    from grounded scene state
-   **Semantic vector fusion** — combining symbolic structure with
    continuous representations
-   **Scene-aware inference** — leveraging grounded object references
    for contextual understanding

Layer-6 is trained on synthetically generated examples derived from
LATN's own parsing and grounding operations, creating a self-supervised
loop where the symbolic system bootstraps neural learning.

### Training Process

Training data is generated on-the-fly by programmatically constructing
spatial reasoning scenarios from the Scene Model. For each scene
configuration, the generator enumerates pairwise object relationships
(above/below, left/right, in front/behind) and produces question-answer
pairs grounded in actual object positions. These natural language pairs
are then processed through LATN Layers 1–5, producing the structural
token sequences and semantic vectors that serve as model inputs.

The training loop uses standard supervised learning with cross-entropy
loss over the output vocabulary. Because training examples are generated
dynamically from LATN's own processing pipeline, the system can produce
~100,000 unique examples per epoch without requiring static dataset
files. This approach ensures the neural model learns representations
that are consistent with LATN's symbolic grounding—the same sentence
parsed by Layers 1–5 during inference will produce structural tokens
identical to those seen during training.

**Output:** Natural language responses grounded in scene state, or
confidence scores for parse disambiguation.
## 2. Grounding in LATN vs. Transformers

A central distinction between LATN and transformer-based models lies in
the mechanism and timing of grounding.

In LATN, grounding is explicit, symbolic, and modular, occurring at
multiple layers as linguistic constructs are mapped onto Scene Objects.
NP grounding in Layer‑2 produces concrete referents in the modeled
environment, enabling deterministic reasoning over objects and spatial
relations.

Transformers encode grounding *implicitly* **(Mikolov et al., 2013)**
through training on large multimodal datasets. Scene elements are
embedded in high-dimensional vector spaces, and grounding emerges via
learned attention patterns---not explicit symbolic reference. While
flexible, this approach lacks transparency and requires massive training
resources.

LATN's hybrid vector‑semantic and symbolic approach provides
transparent, controllable grounding suitable for deterministic domains,
while retaining extensibility for ambiguity resolution.

## 2.1 Scene‑Based Memory and "Infinite Context"

One of LATN's most powerful features---derived from Engraf---is the use
of the *Scene Model as persistent memory.*

Once an NP is grounded to a Scene Object, that referent becomes part of
the world state rather than part of the parse tree. LATN therefore
behaves as if it has an *effectively unbounded context window*, even
though only the *current* sentence is parsed.

Unlike LLMs whose context is bounded by a fixed sequence length, LATN\'s
memory resides in the persistent, evolving vector-space of the Scene
Model, granting it contextual depth *limited only by the knowledge base,
not the input buffer*. LATN simply consults its grounded world state.

**Consequences:** - Object identities persist across sentences - Spatial
and temporal relations remain available - Operations accumulate in a
timeline - Pronouns and anaphora resolve via referents, not text spans -
Memory lives in the *world*, not in the token buffer

This enables Engraf and LATN to implement a form of *world‑model‑based
long‑term memory*, a capability now sought through retrieval‑augmented
LLMs and state-space architectures.

## 2.2 Grounding Mechanisms

LATN employs several grounding mechanisms, deterministic unless
ambiguity demands hypothesis proliferation:

-   **Exact Match Grounding** --- NP vector matches a unique Scene
    Object
-   **Similarity‑Based Grounding** --- multiple Scene Objects match
    semantically; hypotheses branch
-   **Reference‑Frame Grounding** --- PP attaches based on spatial
    feasibility and Scene Object geometry
-   **Timeline Grounding** --- pronouns/adverbs bind to recently
    manipulated objects
-   **Transformation Grounding** --- verbs apply additive/multiplicative
    vector transformations

These incremental grounding operations allow LATN to interpret each
sentence without reprocessing prior discourse.

## 3. Conclusion and Future Directions

The Layered ATN architecture synthesizes Engraf's early vector‑semantic
insights with a modern modular parsing approach. By decomposing language
understanding into sequential layers with explicit grounding, LATN
achieves:

-   **Transparency** --- each layer has interpretable structure
-   **Modularity** --- each transformation is isolated and testable
-   **Grounding** --- meaning connects directly to Scene Objects
-   **Infinite‑Context Behavior** --- memory lives in the world model
-   **Extensibility** --- Layer-6 bridges symbolic and neural approaches

The current Python implementation includes 494+ unit tests validating
lexical analysis, phrase extraction, grounding logic, spatial reasoning,
and end-to-end sentence interpretation. This comprehensive test suite
ensures the modular architecture remains robust as new capabilities are
added.

LATN is not positioned as a replacement for transformer-based models but
as a complementary architecture suitable for *hybrid neuro‑symbolic
systems*.

Future work includes: - declarative and interrogative sentences -
expanded temporal reasoning - agent-style interpretation loops -
multimodal memory

These extensions would bring Engraf's original 1975 vision full circle:
an executable language understanding system grounded in both mathematics
and experience.

## References

Eastman, J. F. (1975). "An N-Space Model for Visual and Verbal
Concepts". PhD Dissertation, North Carolina State University.
<https://github.com/jeffeastman-2/Engraf/blob/main/engraf/An_N_Space_Model/An%20N-Space%20Model%20of%20Visual%20and%20Verbal%20Concepts.pdf>

Woods, W. A. (1970). *Transition network grammars for natural language
analysis.* Communications of the ACM, 13(10), 591--606.

Winograd, T. (1972). *Understanding Natural Language.* MIT Press.

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning
representations by back-propagating errors.* Nature, 323, 533--536.

Vaswani, A., et al. (2017). *Attention is All You Need.* Advances in
Neural Information Processing Systems.

Graves, A. (2012). *Sequence Transduction with Recurrent Neural
Networks.* arXiv:1211.3711.

Plate, T. (1995). *Holographic reduced representations.* IEEE
Transactions on Neural Networks.

Mikolov, T., Sutskever, I., et al. (2013). *Distributed Representations
of Words and Phrases and Their Compositionality.*NeurIPS.

Eastman, J.F. (2025). "Too Early: Reflections on an N-Space Model for
Language and Vision (1975)". <https://zenodo.org/records/15660484>.

Eastman, J. F. (2025). "Vectors are All You Need: A Retrospective".
<https://zenodo.org/records/17806857>.

Eastman, J. F. (2025). "Engraf, a Python Reimplementation with LATN".
<https://github.com/jeffeastman-2/Engraf>

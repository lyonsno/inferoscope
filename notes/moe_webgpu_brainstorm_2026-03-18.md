# MOE / WebGPU visualization brainstorm dump

Raw brainstorm dump copied from chat on 2026-03-18 before formal synthesis.

## What the project is

- Not "make a sexy GPU lava lamp for router activations."
- Not "invent semantic explanations for patterns."
- Primary goal: build something that is interesting enough to hold my attention and honest enough to maybe yield real interpretability value.
- The real hard problem is:
  - generation happens too fast to parse raw expert activations across layers per token,
  - and even a perfect display would not trivially cash out into human-understandable meaning.
- So the job is to find the right visual abstraction layer, not merely to render more data harder.

## What the project is not

- Not trying to infer "functional meaning" directly from token text with an algorithm.
- Not trying to solve interpretability in one grand coherent move.
- Not trying to force semantics early.
- Not trying to make something "beautiful therefore insightful."
- Main epistemic danger is unearned confidence / elegant compression, not glow or sexiness by themselves.

## Core representational insight

Don't visualize the token while it is still happening.

- Strong idea: one-token-behind rendering.
- Wait until the token is fully generated and all routing info for that token is known.
- Then classify / render it with the full event in hand.
- This is not cheating; it's just refusing to build a cognitively worthless "real-time" lie.

Don't try to detect "function" directly.

- Start with identity patterns / routing fingerprints.
- Then compare those against tokens / outputs / contexts later.
- If useful functional patterns emerge, they should emerge through correspondence, not be hard-coded or hallucinated up front.

## Fingerprints / motif structure

There are multiple scales. Don't flatten them into one too early.

At least three useful ledgers:

1. Whole-token fingerprint bins
   - Full-network routing identity for the completed token.
2. Per-layer fingerprint bins
   - Local motif identity per layer.
   - A token can globally match one family while a given layer matches another family better.
3. Token <-> motif contingency ledger
   - Track which tokens, token classes, contexts, etc. correlate with which identity patterns.
   - This is where interesting stuff may emerge later.

Possible later fourth:

4. Short-window / trajectory motif bins
   - Patterns across 2–5 token runs, not just one token.

Important idea:

- The renderer does not have to read from only the whole-token bin.
- It can combine:
  - global token family,
  - local per-layer family,
  - novelty / fit / confidence,
  - and maybe later trajectory state.
- That multi-scale mismatch is not a bug. It's probably where the interesting stuff lives.

## Visual vocabulary

Two color vocabularies is a very strong idea

- Global / whole-token family gets one color vocabulary.
  - Examples: halo, outer tint, background fill, broad regime color, global "faction."
- Local / per-layer motif family gets another color vocabulary.
  - Accent, interior fill, per-layer identity, local active pattern.

This leaves plenty of other channels free for:

- local texture,
- shading,
- timing,
- motion,
- translucency,
- edge treatment,
- persistence trails,
- confidence / ambiguity,
- novelty,
- intensity / commitment,
- instability.

Analogy we liked

- StarCraft faction palette vs player color:
  - global family = broad "Zerg / Terran / Protoss" style identity,
  - local layer motif = the actual local control/accent identity.

That feels structurally right.

## What to render as objects

Don't render "activations." Render motifs / coalitions / commitments

Possible object types:

- recurring expert coalitions,
- recurring whole-token routing identities,
- recurring per-layer routing identities,
- commitment / confidence / entropy shifts,
- novelty vs repetition,
- transitions between motif regimes.

Very important:

- Meaning comes later, maybe.
- First win is just:
  - can recurring identity structure become visible and stable enough to be treated as a real object?

## Anti-bullshit instrumentation

This is not optional.

Because the danger is not glow.
The danger is thinking "I know what this is" before I actually do.

Must-have anti-delusion handles

1. Raw substrate view
   - For any rendered pattern, I should be able to inspect the underlying fingerprint / routing basis easily.
   - Not necessarily always on-screen, but quickly accessible.
2. Replay
   - Same prompt / same token / same run conditions should be replayable.
   - If the motif is real enough to matter, it should recur in some recognizable way.
3. Null / scramble / matched-fake mode
   - Feed the renderer shuffled / distribution-matched fake routing and see how much "there" remains.
   - Good for calibrating whether I'm seeing structure or getting high on my own abstraction.
4. Confidence / fit / ambiguity display
   - If a token weakly matches a motif, show that.
   - Don't let all assignments look equally certain.
5. Comparison support
   - Token A vs token B.
   - Same token across different contexts.
   - Same motif across different prompts / runs.
   - Local vs global mismatch.
6. Novelty channel
   - Make "not like prior known shit" visible.
   - Don't force every event into an existing box.

Main warning, restated properly

- The danger is mistaking a coherent representational trick for a coherent epistemic object.
- Elegant compression is the real siren song, not the sexy shader.

## 2D vs 3D

Important correction

- Not "2D honest, 3D lying whore."
- Both 2D and 3D can support pareidolia and bullshit.
- The issue is not dimensionality by itself.
- The issue is:
  - auditability,
  - comparability,
  - and whether the representation stabilizes real objects or just vibes.

Where we landed

2D / schematic / layer-stack family

- Better audit surface in many cases.
- Easier for:
  - debugging,
  - comparison,
  - side-by-side token inspection,
  - checking bin coherence,
  - seeing whether assignments are bullshit.

Possible forms:

- simple layer rows,
- curved rows,
- tendrils / fingers,
- stacked strips,
- flattened circuit-board vibe.

3D / stacked discs / tessellated slices family

- Probably not clearer early, and may be a bit less clear at first.
- But if clustering / motifing actually cashes out strongly, 3D could become very, very good.
- Especially useful if:
  - local neighborhoods become meaningful,
  - experts with similar motif membership can be laid out contiguously,
  - depth helps express persistence / weather / relationship between layers.

Best current read

- 3D is not inherently more deceptive.
- It may just be less auditable early and easier to over-endow with objecthood.
- But if it is the version that will actually get built and keep me engaged, that matters a lot.
- So:
  - build the thing that holds me,
  - but with anti-bullshit handles strapped to it.

Third visual family that should stay in the hypothesis space

- Flow / transition / alluvial / braid / stream view
- Not "where are the experts on each layer," but "how does motif-state move through the stack / over token time."
- Probably not first.
- But worth keeping alive as a later view.

## Strong staging idea

Build in phases

1. Version 0: ugly honest object
   - One-token-behind.
   - Basic fingerprint extraction.
   - Very simple rendering.
   - Just enough to see what kind of confusion appears.
2. Version 1: motif bins
   - Whole-token bins.
   - Per-layer bins.
   - Color vocabularies.
   - Confidence / novelty.
3. Version 2: comparative / audit tooling
   - Replay.
   - Null mode.
   - Raw substrate access.
   - Side-by-side comparison.
4. Version 3: geometry upgrades
   - Reordering experts spatially based on actual bin/co-occurrence structure.
   - Maybe then 3D if it has earned the right.
5. Version 4: flow / transition views
   - Only once motifs exist well enough to transition between.

Very important staging principle

- Don't force coherence.
- Let the data refuse coherence if it wants to.
- Avoid coherence like the plague unless the data leaves no alternative.

## Specific geometric idea that seems promising

Stable expert geography by bin proximity

- If the bins turn out to be strong enough:
  - lay out experts per layer so that same-bin / commonly co-active experts are closer together.
- Then the visualization becomes less random.
- If that works, 3D stacked slices get much more appealing.
- If it doesn't, fine — use boring fixed indices and keep moving.

Important caveat

- Don't make success depend on this working beautifully from the start.
- First version can use fixed dumb positions.

## "Commitment" as an interesting derived object

This came up and seems good:

- Don't just show who lit up.
- Show where routing feels:
  - diffuse,
  - committed,
  - unstable,
  - switching,
  - persistent.
- Commitment / entropy / sharpness of routing may be more human-legible than raw activation patterns.

Not a must-have in version 1, but a real later dimension.

## Working philosophical stance

- Build for my own attention first.
- If it becomes useful to others later, translate later.
- This is not a public demo first.
- I do not need to protect myself from sexy graphics like some peasant seeing electricity.
- I do need to protect myself from my own ability to mistake elegant compression for actual knowledge.
- So:
  - sexy is fine,
  - beautiful is fine,
  - hypnotic is fine,
  - as long as the instrument can still be cross-examined.

## Current best one-sentence project brief

Build a one-token-delayed WebGPU instrument that turns recurring MoE routing identities into global and local motif objects, keeps meaning bracketed, and remains auditable enough not to become a self-licking interpretability shrine.

## Current best "where we landed"

- One-token-behind rendering: yes, strong.
- Identity motifs before semantics: yes, strong.
- Whole-token + per-layer ledgers: yes.
- Token/motif contingency table: yes.
- Two color vocabularies: yes, very strong.
- Null/replay/confidence/raw-substrate: yes, necessary.
- Avoid coherence unless forced by data: yes.
- 2D vs 3D:
  - not a morality play,
  - 2D often better for audit early,
  - 3D may be excellent later if the motifs earn it,
  - and if 3D is what actually gets built, that may still be the correct first move.
- Main epistemic danger: elegant compression / unearned confidence, not prettiness alone.
- Likely next step: build the first ugly thing and let it teach you what kind of confusion you actually have.

## Brutally short clipboard version

### Project

- WebGPU visualization of MoE expert routing during generation.
- Render token one step behind so full routing for token is known before display.
- Don't infer semantics first; build identity/routing motifs first.

### Core structures

- Whole-token fingerprint bins.
- Per-layer fingerprint bins.
- Token <-> motif contingency ledger.
- Maybe later short-window/trajectory bins.

### Rendering

- Two color vocabularies:
  - global token family,
  - local layer motif family.
- Leave motion/texture/shading/timing/translucency/etc. free for other signals.
- Don't force one grand coherent representation.

### Anti-bullshit

- Raw substrate view.
- Replay.
- Null/scramble mode.
- Confidence/ambiguity.
- Novelty.
- Comparison across tokens/runs.

### 2D vs 3D

- 2D often better early for audit/debug/compare.
- 3D may be excellent later if motifs/clustering earn it.
- Sexiness is not the problem; elegant compression / false confidence is.
- Build the version that holds attention, but keep it interrogable.

### Big principle

- Avoid coherence unless the data gives no alternative.
- First win is "right confusion," not final meaning.

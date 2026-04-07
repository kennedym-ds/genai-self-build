# 🛸 Zara Crash-Lands in Shape World

## Full 20-Minute TED-Style Keynote Script

**Speaker:** Michael Kennedy
**Event:** Analog Layout Conference
**Duration:** 20 minutes (no Q&A baked in — leave 5 mins after if desired)
**Format:** Widescreen (16:9), single-speaker, TED-style delivery

---

### Delivery Notes

- **Pace:** ~130 words/minute (conversational TED pace). Total ≈ 2,600 words.
- **Slides:** Advance on `[SLIDE]` cues. Slides are minimal — you are the show.
- **Demos:** Four live Streamlit demos are embedded. Have the app running on a second screen.
- **Pauses:** Written as `[beat]` (1–2s) or `[long pause]` (3–4s). Don't rush these.
- **Movement:** Start center. Move left for "Zara's perspective," right for "the tech reveal."
- **Voice:** Conversational, curious, warm. Not lecturing — *sharing a story.*

---

## COLD OPEN (0:00 – 1:30)

`[SLIDE 1: Black screen. No title yet.]`

*(Walk to center stage. No slides. No clicker. Just you and the audience.)*

I want to try something a little unusual. I need you to do me a favour.

Close your eyes.

*(Wait. Let the room settle. 3 seconds of silence.)*

I'm serious — close them. Just for a moment.

*(2 more seconds.)*

Now — open.

`[SLIDE 2: Title slide — "Zara Crash-Lands in Shape World"]`

That flood you just felt — that instant blast of colour, edges, depth, faces — your brain just processed about **eleven million bits** of visual information. In a fraction of a second. Without trying.

Today, I'm going to tell you the story of someone who couldn't do that. Someone who had to learn to see from absolute zero. And in doing that, she accidentally built every component of modern generative AI.

[beat]

Her name is Zara. She's an alien. And she's about to explain AI better than any textbook ever has.

---

## ACT 1: THE CRASH (1:30 – 5:30)

### Scene 1: A World Without Sight

`[SLIDE 3: "A World of Pure Sound" — dark slide, sound wave symbols]`

Zara comes from Zorath-7 — a planet with no light. None. Her species evolved in permanent darkness. They "see" the world through sound: echolocation, vibration frequencies, harmonic patterns. They're brilliant scientists. They have mathematics, engineering, even art — all built on sound.

But they have zero concept of edges. No idea what a corner is. No understanding of colour.

[beat]

Now imagine Zara crash-lands on Earth. The impact cracks open her ship. And for the first time in her life — light enters.

`[SLIDE 4: Explosion of chaotic, colourful shapes — visual overwhelm]`

*(Point at the screen.)*

And she sees... *this*.

[long pause — let the audience sit with the chaos]

Every pixel. Every colour. Every edge. All at once. It's like trying to understand a symphony by hearing every instrument, every note, every harmonic simultaneously with no concept of music.

She is completely, absolutely overwhelmed.

### Scene 2: Three Attempts at Understanding

`[SLIDE 5: "Attempt 1 — Every Pixel" — zoomed grid]`

So what does a scientist do? She experiments.

**Attempt one:** Look at every single pixel. One at a time. Scan left to right, top to bottom. It's a grid of tiny colour values. She catalogues each one.

The result? A hundred numbers for a simple house shape. Technically complete — she hasn't missed a single pixel. Practically useless. It's like trying to understand a song by measuring one air molecule's vibration at a time.

`[SLIDE 6: "Attempt 2 — Memorise Whole Shapes" — shape catalogue]`

**Attempt two:** Go the other direction. Memorise *entire* shapes. "That's a circle. That's a rectangle. That's a triangle." She builds a catalogue.

Works beautifully — until she sees something she's never catalogued.

*(Adopt Zara's panicked voice:)* "What IS that? It's got five sides but one is curved and there's a notch in it — it's not in my catalogue! DOES NOT COMPUTE!"

`[SLIDE 7: "Attempt 3 — Learn the PARTS" — shape decomposition]`

**Attempt three** — and this is the breakthrough: learn the **parts**.

Corners — they repeat. Edges — they repeat. Curves — they repeat. So she builds an alphabet. An alphabet of shape-parts. And suddenly she can describe *any* shape, even brand-new ones, as a combination of parts she already knows.

A hundred tokens for the pixel approach. Three for the whole-shape approach — but fragile. Twelve for the parts approach — flexible, efficient, and it handles novelty.

`[SLIDE 8: "This is Tokenization" — side-by-side comparison]`

This is **tokenization**. And this is exactly what happens when you upload a photo to ChatGPT. The image gets chopped into sixteen-by-sixteen patches. Each patch becomes a token — a "shape word." That's Vision Transformer tokenization. And it's how text models work too — "unbreakable" becomes "un" + "break" + "able." Three known parts, infinite combinations.

[beat]

Zara just reinvented something it took humans decades to figure out. She's good at this.

---

## ACT 2: THE MAP (5:30 – 10:00)

### Scene 3: Numbers Without Meaning

`[SLIDE 9: Act 2 title — "Do These Go Together?"]`

So Zara can break shapes into parts now. She's assigned each part a number. Edge-type-42. Edge-type-43. Neat and tidy.

But there's a problem.

On her number line, a sharp corner and a gentle curve are one number apart. A tiny circle and a massive square are two apart. The numbers tell her nothing about what the shapes actually *are*.

`[SLIDE 10: The Meaning Map — 2D scatter plot]`

She needs something better. She needs a **map**.

So she builds one. She arranges shapes in space so that similar shapes live *close together* and different shapes live *far apart*. Circles cluster over here. Rectangles cluster over there. And triangles — they sit somewhere in between, because they share properties with both.

But here's where it gets magical.

*(Lean in.)*

The *directions* on the map have meaning.

If she knows that "triangle on top of rectangle" means "house"... she can walk in that same direction from "dome" and land on "mosque." She can walk from "small circle" toward "small square" and discover that the same journey from "large circle" gets her to "large square."

`[SLIDE 11: "Meaning = Distance" — embedding analogies]`

The map isn't just organising shapes. It's capturing **relationships**. The direction from one shape to another *is* the relationship.

This is what AI researchers call **embeddings**. Every shape — every token — becomes a point in multi-dimensional space. And "meaning" becomes... distance.

[beat — let that land]

### Scene 4: The Shape Library

`[SLIDE 12: "The Shape Library" — vector DB concept]`

Now Zara runs into her next problem. She's been exploring Earth for weeks. She's collected *thousands* of shape arrangements. Houses, trees, faces, vehicles, circuit-board-looking things she found in someone's lab.

When she sees something new, she wants to say: "Have I seen something *like* this before?"

She doesn't know the name of what she's looking at. She can't search by label. She needs a library where you search by **similarity**.

`[SLIDE 13: Vector database visualisation]`

So she builds one. You drop in a query — a shape, a scene, a pattern — and the library finds the nearest neighbours. Not by matching keywords. By measuring *distance* in the meaning map.

*(Step toward audience.)*

Quick show of hands — how many of you have ever flipped through old tape-outs, looking for something "kind of like" what you're working on now? Where you sort of remember it but you can't quite find it?

*(Pause for hands.)*

Congratulations. You're a **human vector database**.

*(Let them laugh.)*

`[SLIDE 14: Real-world vector DB examples]`

And if you've used Google Photos and it magically groups all photos of the same person — that's a vector database. Spotify's "Discover Weekly"? Vector database. GitHub's code search? Vector database.

Zara's little shape library is the same architecture behind all of them.

---

## ACT 3: THE SPOTLIGHT (10:00 – 14:30)

### Scene 5: Context Is Everything

`[SLIDE 15: Act 3 title — "Wait, Context Matters!"]`

Zara's getting talented. She can identify shapes, map meaning, search her library. She's feeling good about herself.

And then she makes a mistake.

`[SLIDE 16: Red circle in two contexts — traffic light vs clown nose]`

She sees a red circle. She calls it "red circle." Accurate. But she calls it the same thing when it's on a traffic light and when it's on a clown's nose.

Same shape. Completely different meaning.

*(Pause. Let the visual land.)*

She doesn't get that what's *around* a shape changes what the shape *means*. She's looking at every shape in isolation. And isolation is where meaning goes to die.

### Scene 6: The Spotlight System

`[SLIDE 17: Spotlight system — attention concept]`

So she invents something brilliant. A spotlight system.

Every shape gets to shine a spotlight on every *other* shape and ask one question: "Are you important to me?"

And she doesn't use just one spotlight. She uses four. Each looking for a different kind of relationship:

- **Proximity:** "Are you close to me?"
- **Colour match:** "Do we look alike?"
- **Alignment:** "Are we lined up?"
- **Containment:** "Am I inside you — or are you inside me?"

`[SLIDE 18: Multi-head attention heatmap]`

When she looks at a triangle sitting on a rectangle, the alignment spotlight goes *crazy* between them — it screams "THESE ARE CONNECTED!" The colour spotlight barely fires. The proximity spotlight fires a bit. And the containment spotlight notices the windows inside the walls.

Each spotlight sees a different relationship. Combined, they see *everything*.

`[SLIDE 19: "This is Attention"]`

This is **multi-head self-attention**. It's the breakthrough from 2017 — the paper called "Attention Is All You Need" — and it's the reason we suddenly went from AI that was *pretty good* at language to AI that was *shockingly good* at language.

It's how ChatGPT knows that "it" in "The cat sat on the mat because **it** was tired" refers to the cat, not the mat. Every word shines a spotlight on every other word.

### Scene 7: The Artist

`[SLIDE 20: "Can I CREATE Something New?"]`

Now Zara has all the pieces.

A way to see parts — tokenization.
A map of meaning — embeddings.
A library of patterns — vector database.
Spotlights for context — attention.

She asks the big question: *Can I create something new?*

She stacks all her tools into one system. Feed in some shapes. The system predicts: what shape goes next? And where should it go? One shape at a time — like writing a sentence, word by word.

`[SLIDE 21: Zara's first creation — hilariously bad]`

Her first attempt is... not great.

*(Pause for laugh.)*

But the architecture — the architecture is right. It's the same architecture behind DALL-E, behind Stable Diffusion, behind ChatGPT.

`[SLIDE 22: Scale comparison — 10K to 1.8T parameters]`

The difference?

Zara's toy model has about ten thousand parameters. Cute.
DALL-E 3 has three billion. Impressive.
GPT-4 has over a trillion. Mind-boggling.

Same engine. Enormously different horsepower.

It's like comparing a go-kart to a Formula One car. Same principles of physics. Very different lap times.

---

## ACT 4: THE SAFETY NET (14:30 – 17:00)

### Scene 8: When Imagination Goes Wrong

`[SLIDE 23: Act 4 title — "Check Your Work"]`

Zara has one last problem. And it's a big one.

Sometimes her generator... makes things up. Convincingly. She creates a stop sign with five sides. A face with the mouth above the eyes. It looks *plausible* — until you actually look at it.

`[SLIDE 24: Hallucination examples — bad stop sign, wrong face]`

Sound familiar? This is the hallucination problem. And it happens for the same reason a student who studied hard but didn't bring their notes to the exam sometimes writes confidently wrong answers. The knowledge is approximate. The recall is imperfect. And the system doesn't *know* that it doesn't know.

### Scene 9: Look Before You Leap

`[SLIDE 25: RAG pipeline — Search → Retrieve → Generate]`

Zara's solution is **brilliantly humble**.

Before creating something new — *look at what already exists*. Search the library. Pull up similar examples. Use them as a guide. *Then* generate.

Don't trust your memory alone. **Check your work.**

This is called **RAG** — Retrieval-Augmented Generation. It's the reason ChatGPT can answer questions about today's news even though its training data is months old. It looks things up first.

And it's the reason AI for **your** world doesn't have to hallucinate layouts from scratch. It can search your proven designs — layouts that passed DRC, that survived LVS, that taped out successfully — and use them as starting points.

---

## THE REVEAL (17:00 – 20:00)

### Scene 10: The Mirror

`[SLIDE 26: Zara's Complete Journey — six icons in sequence]`

*(Move to centre. Slow down.)*

Let me take you back through Zara's journey. One more time.

She learned to **break images into pieces** — that's tokenization.
She **mapped meaning into space** — embeddings.
She built a **searchable library** — a vector database.
She learned **which relationships matter** — attention.
She assembled a **generator** — a transformer.
And she learned to **check her work** — RAG.

[beat]

`[SLIDE 27: Side-by-side — Shapes vs Chip Layout]`

Now here's the reveal.

Replace "shapes" with **polygons on metal layers**.
Replace "arrangements" with **layout floorplans**.
Replace "library" with **your corporate IP database**.
Replace "parts" with **standard cell primitives**.

*(Pause. Let the connection land.)*

The math is identical. The concepts are identical. The only difference is what the shapes represent.

`[SLIDE 28: Industry table — who's doing what]`

And this isn't theoretical. Google published in *Nature* in 2021 using reinforcement learning for TPU chip floorplanning. Cadence has ML-assisted placement in Virtuoso. Synopsys has ML constraint extraction in Custom Compiler. DARPA's ALIGN project generates analog layout directly from netlists. UT Austin's MAGICAL project automates analog layout generation.

It's not science fiction. It's happening **right now**, in tools some of you are already using.

### Scene 11: The Last 20%

`[SLIDE 29: "Your Copilot, Not Your Replacement"]`

*(Step forward. Personal.)*

I know what some of you might be thinking. "Great talk, Michael. Now tell me honestly — am I going to be replaced by a robot?"

[beat]

No. And here's why.

`[SLIDE 30: "The Last 20%"]`

Zara learned to see patterns. But she never learned **taste**. She doesn't know *why* you put those guard rings there. She doesn't feel the intuition that says "that parasitic is going to bite us at high frequency." She doesn't have twenty years of tribal knowledge about what works at your foundry, with your process corners, with your team.

AI is the most talented intern you've ever had. It remembers every layout. It searches by similarity in milliseconds. It can suggest a first-pass placement that's eighty percent there.

The last twenty percent? That's **you**. That's the art. That's the hard-won intuition that no amount of training data can replicate.

And *that* is not going anywhere.

### Scene 12: Three Takeaways

`[SLIDE 31: Three Takeaways]`

Three things to take with you.

**One:** AI learns to see shapes the same way it learns to read text. Break it apart, map meaning, search, attend, generate, verify. It's the same pipeline, whether you're processing Shakespeare or silicon.

**Two:** This is already happening in chip design. Not in ten years. Now. In tools you're evaluating today.

**Three:** AI is your copilot, not your replacement. Patterns are learnable. Taste is not.

### Scene 13: Close

`[SLIDE 32: Closing slide — Thank You]`

*(Return to centre. Calm. Warm.)*

Zara went from not knowing what a shape was... to being able to generate and verify arrangements. In twenty minutes.

That's the generative AI journey. And now it's yours too.

Thank you.

*(Hold. Don't rush off. Let the applause find you.)*

---

## TIMING REFERENCE

| Section | Start | End | Duration |
|---------|-------|-----|----------|
| Cold Open | 0:00 | 1:30 | 1:30 |
| Act 1: The Crash (Tokenization) | 1:30 | 5:30 | 4:00 |
| Act 2: The Map (Embeddings + Vector DB) | 5:30 | 10:00 | 4:30 |
| Act 3: The Spotlight (Attention + Transformers) | 10:00 | 14:30 | 4:30 |
| Act 4: The Safety Net (RAG) | 14:30 | 17:00 | 2:30 |
| The Reveal & Close | 17:00 | 20:00 | 3:00 |
| **TOTAL** | | | **20:00** |

## REHEARSAL TIPS

1. **Practice the cold open with real silence.** The 5 seconds of closed eyes feel eternal. That's the point.
2. **Know the Zara voice.** When you speak *as* Zara ("DOES NOT COMPUTE!"), commit. The audience will love it.
3. **The "human vector database" moment** is your biggest reliable laugh. Let it land. Don't talk over the laughter.
4. **The reveal (shapes → chip layout)** is the intellectual climax. Slow down here. Let the audience make the connection before you spell it out.
5. **The "last 20%" section** is where the room exhales. They've been worrying about job replacement since you started. Address it directly and with conviction.
6. **End clean.** "Thank you" — full stop. No "so yeah," no "that's about it." Just "Thank you."

# Zara Crash-Lands in Shape World

### 20-Minute TED-Style Keynote

**Created**: 2026-03-06  
**Status**: Draft → Awaiting Approval  
**Author**: Michael Kennedy (michael.kennedy@analog.com)  
**Target Event**: Physical Layout Conference  
**Format**: 20-minute keynote, TED-style — light, narrative-driven, visual, minimal jargon  

---

## Abstract

> An alien scientist from a world of pure sound crash-lands on Earth — and opens her eyes for the first time. No edges. No corners. No color. Just noise. Through Zara's journey to understand shapes, we'll trace the six ideas that teach any AI to see: breaking images into parts, mapping meaning into space, building searchable memory, learning what to focus on, generating new arrangements, and knowing when to check your work. By the end, you'll realize these aren't abstract AI concepts — they're the exact techniques being applied to semiconductor layout today. Light on jargon. Heavy on "aha" moments. No prior ML knowledge required — just bring your curiosity and your skepticism.

---

## TL;DR

A 20-minute story about an alien named Zara who has **never seen a shape** — she comes from a world of pure sound. She crash-lands on Earth, opens her eyes, and has to learn what shapes are, how they relate, and eventually how to *create* new arrangements. As the audience follows her journey, they accidentally learn the six core concepts behind how GenAI understands images — from tokenization to retrieval-augmented generation. The final reveal: *this is exactly how AI could learn to lay out your circuits.*

**Tone**: TED talk. Funny. Visual. No equations on screen. The audience should laugh, nod, and leave thinking "I actually get how this works now."

---

## The Talk in One Sentence

> "What if an alien who'd never seen a shape had to learn to design semiconductor layouts?"

---

## Zara's Story (The Entire Narrative)

### Opening Hook (0:00 – 1:30)

> *"I want you to imagine something impossible. Imagine you've never seen a shape. Not a circle. Not a square. Not a line. Nothing.*
>
> *You come from a world of sound. You perceive the universe through vibrations, frequencies, echoes. You're brilliant — a scientist, in fact. But you have no concept of 'edges' or 'corners' or 'color.'*
>
> *Now imagine you crash-land on Earth. You open your eyes for the first time. And you see... this."*
>
> **[SLIDE: Chaotic collage of colorful geometric shapes — circles, rectangles, triangles, polygons, overlapping, nested, scattered]**
>
> *"Meet Zara. She's an alien. And she's about to go on the exact same journey that every AI goes on when we teach it to understand images."*

**Why this works**: The audience immediately empathizes. They're visual thinkers — they can't imagine NOT seeing shapes. It's disorienting. That disorientation IS the point.

---

### Act 1: "What Am I Looking At?" (1:30 – 5:00)

**GenAI concept woven in: Tokenization + Early Vision**

> *"Zara's first problem: she's overwhelmed. There's TOO MUCH to look at. Every pixel, every color, every edge — it's all noise. She needs to break what she sees into manageable pieces."*
>
> **[SLIDE: Same chaotic image, then an animated overlay showing it being diced into a grid of patches]**
>
> *"Her first attempt: look at every single pixel. One at a time. It's like trying to understand a song by listening to one air molecule vibrating at a time. Technically complete. Practically useless."*
>
> *"Her second attempt: memorize every complete shape. 'That's a circle! That's a rectangle!' Works great — until she sees a shape she's never seen before. 'What IS that?! It's not in my catalog!'"*
>
> *"Her breakthrough: learn the PARTS. Corners repeat. Edges repeat. Curves repeat. She builds a alphabet of shape-parts — and suddenly she can describe ANY shape, even brand new ones, as a combination of parts she already knows."*

**[SLIDE: Visual showing: raw image → pixel grid (huge!) → shape catalog (rigid!) → learned shape-parts (flexible!)]**

> *"By the way — this is exactly what happens when you upload a photo to ChatGPT. The image gets chopped into patches. Each patch becomes a token — a 'shape-word.' That's all tokenization is: turning something continuous into manageable pieces."*

**Light touch**: No math. No vocabulary. Just the visual progression from overwhelming to organized.

---

### Act 2: "Do These Go Together?" (5:00 – 9:00)

**GenAI concepts woven in: Embeddings + Vector Search**

> *"Zara can identify shape-parts now. But she has a problem. She gave each part a number. Edge-type-42. Edge-type-43. But the numbers are meaningless — a sharp corner and a gentle curve are one number apart. She needs a way to capture that some shapes are SIMILAR and others are DIFFERENT."*
>
> *"So she builds a map."*

**[SLIDE: 2D scatter plot — clusters of shapes forming visible groups. Circles near circles. Angular shapes near angular shapes. A "house" emoji near "building" shapes.]**

> *"On this map, similar shapes live close together. Different shapes live far apart. And here's the magical part — the map captures RELATIONSHIPS. If she knows that a 'triangle on top of a rectangle' means 'house'... she can figure out that a 'dome on top of a rectangle' might mean 'mosque.' The DIRECTION between shapes has meaning."*
>
> *"This is what AI researchers call embeddings. Every shape becomes a point in space. And 'meaning' becomes... distance."*

**[Beat — let that land]**

> *"Now Zara has another problem. She's collected THOUSANDS of these shape arrangements. When she sees something new, she wants to say 'Have I seen something like this before?' She needs a library — one where you don't search by name, you search by SIMILARITY."*
>
> *"You know that moment when you're browsing your phone photos and Google says 'These look similar'? That's a vector database. It embeds your photo into numbers, then finds the nearest neighbors."*

**[SLIDE: "Shape Library" — query shape on left, top-3 similar results on right with similarity scores]**

> *"Raise your hand if you've ever flipped through old tape-outs looking for something 'kind of like' what you're working on now."*

**[Pause for laughs and hands]**

> *"Congratulations — you're a human vector database."*

---

### Act 3: "Wait, Context Matters!" (9:00 – 13:00)

**GenAI concepts woven in: Attention + Transformers**

> *"Zara's getting good. She can identify shapes, know which ones are similar, and search her library. But she keeps making a mistake."*
>
> *"She sees a red circle. Fine — she knows circles. But she treats it the same whether it's on a traffic light or on a clown's nose. She doesn't get that the SAME shape means completely different things depending on what's AROUND it."*
>
> *"She needs to learn what humans do effortlessly: pay attention to context."*

**[SLIDE: Same red circle shown in two contexts — traffic light vs. clown face. Caption: "Same shape. Very different meaning."]**

> *"So Zara invents a spotlight system. Every shape gets to shine a spotlight on every other shape and ask: 'Are you important to me?'"*
>
> *"And here's the cool part — she doesn't use ONE spotlight. She uses four. One looks for shapes that are close together. One looks for shapes the same color. One looks for shapes that are aligned. One looks for shapes that contain each other."*

**[SLIDE: Animated attention heatmap — click a shape, see which other shapes light up]**

> *"When she looks at a triangle sitting on top of a rectangle, the 'alignment' spotlight goes crazy between them. When she looks at two matching triangles in a symmetric pattern, the 'proximity and color' spotlights both fire."*
>
> *"This is attention — the breakthrough that transformed AI in 2017. It's how the machine learns that 'it' in 'The cat sat on the mat because IT was tired' refers to the cat, not the mat. And it's how vision AI learns that two shapes RELATE to each other across an image."*

**[Transition beat]**

> *"Now Zara has all the pieces. A way to see parts. A map of meaning. A library of patterns. Spotlights for context. She asks the big question:"*
>
> *"Can I CREATE something new?"*

**[SLIDE: Zara at a workbench with all her tools. Caption: "The Artist."]**

> *"She stacks all her tools into one system. Feed in some shapes — the system predicts: what shape goes next, and WHERE should it go? One shape at a time, like writing a sentence word by word."*
>
> *"Her first attempt is... not great."*

**[SLIDE: Hilariously bad shape arrangement — house with roof underground, tree growing sideways]**

**[Pause for laughs]**

> *"But the architecture is right. It's the same architecture behind DALL-E, Stable Diffusion, and ChatGPT. The difference? Our little demo has ten thousand parameters. GPT-4 has over a trillion. Same engine — enormously more horsepower."*

---

### Act 4: "Check Your Work" (13:00 – 16:00)

**GenAI concept woven in: RAG (Retrieval-Augmented Generation)**

> *"Zara has one last problem. Sometimes her generator hallucinates. It creates shapes that look plausible but are nonsensical. A five-sided stop sign. A face with the mouth above the eyes."*
>
> *"Her solution is brilliantly humble: before creating something new, LOOK at what already exists. Search the library. Pull up similar examples. Use them as a guide. THEN generate."*

**[SLIDE: Pipeline animation: Request → Search library → Retrieve 3 examples → Generate (guided by examples) → Output with citations]**

> *"This is called RAG — Retrieval-Augmented Generation. It's the reason ChatGPT can answer questions about today's news even though it was trained months ago. It looks things up first."*
>
> *"And it's the reason AI for YOUR world doesn't have to hallucinate layouts from scratch. It can retrieve proven designs from your own library — layouts that passed DRC, that taped out successfully — and suggest adaptations."*

---

### The Reveal: "Sound Familiar?" (16:00 – 18:30)

> *"So let me recap Zara's journey."*

**[SLIDE: Clean visual — six icons in a line with one-word labels]**

> *"She learned to break images into pieces. She mapped meaning into space. She built a searchable library. She learned which relationships matter. She assembled a generator. And she learned to check her work against known examples."*
>
> *"Now... replace 'shapes' with 'polygons on metal layers.' Replace 'arrangements' with 'layout floorplans.' Replace 'library' with 'your corporate IP database.'"*

**[SLIDE: Side-by-side — Zara's colorful shape world on the left, a real layout view on the right. They look strikingly similar.]**

> *"The math is identical. The concepts are identical. The only difference is what the shapes represent."*
>
> *"Companies like Cadence, Synopsys, and Google are already applying these exact techniques. Google published a paper in Nature using AI for chip floorplanning. Cadence and Synopsys have ML-assisted placement in their latest tools. Research projects like ALIGN and MAGICAL are generating analog layouts from netlists."*
>
> *"It's not science fiction. It's happening."*

---

### Closing: "Your Copilot, Not Your Replacement" (18:30 – 20:00)

> *"I know what some of you are thinking. 'Great. I'm going to be replaced by a robot.'"*
>
> *"You're not. And here's why."*
>
> *"Zara learned to see patterns. But she never learned TASTE. She doesn't know why you put those guard rings there. She doesn't feel the intuition that says 'that parasitic is going to bite us.' She doesn't have twenty years of tribal knowledge about what works at your foundry."*
>
> *"AI is the most talented intern you've ever had. It remembers every layout you've ever done. It can search by similarity in milliseconds. It can suggest a first-pass placement that's 80% there."*
>
> *"The last 20%? That's you. That's the art. And that's not going anywhere."*

**[Final slide: Zara waving goodbye, surrounded by shapes. Caption: "Same math. Different shapes. Your tape-out."]**

> *"Zara went from not knowing what a shape was... to being able to generate and verify arrangements. That's the GenAI journey, in twenty minutes."*
>
> *"Thank you."*

---

## Timing Summary

| Section | Duration | Content |
|---|---|---|
| **Opening Hook** | 1:30 | Meet Zara, the shape-blind alien |
| **Act 1**: "What am I looking at?" | 3:30 | Tokenization — breaking images into parts |
| **Act 2**: "Do these go together?" | 4:00 | Embeddings + Vector search — meaning as distance |
| **Act 3**: "Context matters!" | 4:00 | Attention + Transformers — relationships and generation |
| **Act 4**: "Check your work" | 3:00 | RAG — retrieve before you create |
| **The Reveal** | 2:30 | Shapes → layout. The bridge moment. |
| **Closing** | 1:30 | Copilot, not replacement. |
| **Total** | **20:00** | |

---

## Slide Deck Estimate: ~25-30 slides

| # | Slide | Type |
|---|---|---|
| 1 | Title: "An Alien Visits Shape World" | Title |
| 2 | "Imagine you've never seen a shape" | Text-on-black (dramatic) |
| 3 | Chaotic colorful shape collage | Full-bleed image |
| 4 | Meet Zara — character card | Illustration |
| 5 | "Too much to look at" — pixel overwhelm | Animation/image |
| 6 | Pixel grid decomposition | Diagram |
| 7 | Shape catalog (works but brittle) | Diagram |
| 8 | Learned shape-parts (BPE equivalent) | Diagram |
| 9 | "This is tokenization" — ChatGPT connection | Clean text |
| 10 | Zara's meaning map — 2D scatter | Interactive/static viz |
| 11 | Shape clusters forming | Animated scatter |
| 12 | "Meaning = distance" | Quote slide |
| 13 | Shape library — query + results | Demo screenshot |
| 14 | "You're a human vector database" | Laugh line |
| 15 | Red circle — two contexts | Side-by-side |
| 16 | Spotlight system — attention concept | Diagram |
| 17 | Four spotlights (multi-head) | Icons + labels |
| 18 | Attention heatmap demo | Demo screenshot |
| 19 | "Can I CREATE?" — Zara at workbench | Illustration |
| 20 | Bad generation — hilarious output | Funny image |
| 21 | "Same architecture, different scale" table | Comparison |
| 22 | RAG pipeline — retrieve then create | Flow diagram |
| 23 | "It looks things up first" | Clean text |
| 24 | Zara's journey recap — six icons | Icon row |
| 25 | Side-by-side: Shape World vs. layout | Split image |
| 26 | Industry names — Cadence, Synopsys, Google | Logos + citations |
| 27 | "Your copilot, not your replacement" | Quote slide |
| 28 | The art is the last 20% | Clean text |
| 29 | Zara waving goodbye | Closing illustration |
| 30 | "Thank you" + contact info | End card |

---

## What Changes in the Repo

### Lighter Scope (20-min talk = demo support, not full workshop modules)

The goal is **a single Streamlit app** that powers 3-4 live demo moments during the talk + the slide deck. We do NOT need six full workshop rewrites.

```
genai-self-build/
├── workshops/              ← UNTOUCHED
├── keynote/                ← NEW
│   ├── app.py              # Single Streamlit demo app (all 4 demo moments)
│   ├── shape_data.py       # Shape definitions, sample patterns, color palette
│   ├── shapes_core.py      # Lightweight: tokenizer + embeddings + vector DB + attention
│   │                       # (one file, not six — just enough to power the demos)
│   ├── requirements.txt    # streamlit, numpy, matplotlib
│   └── slides/
│       └── keynote.md      # Marp slide deck (~30 slides)
```

### The Four Demo Moments

| Demo | When in Talk | What It Shows | Powered By |
|---|---|---|---|
| **Shape Tokenizer** | Act 1 (~3:00) | Canvas of shapes → decomposed into parts three ways | `shapes_core.py` tokenizer |
| **Meaning Map** | Act 2 (~6:00) | 2D scatter plot of shape embeddings, clusters visible | `shapes_core.py` embeddings |
| **Shape Library Search** | Act 2 (~8:00) | Pick a shape → find similar arrangements | `shapes_core.py` vector DB |
| **Attention Heatmap** | Act 3 (~10:00) | Click shape → see attention to other shapes, toggle heads | `shapes_core.py` attention |

Transformer generation and RAG are **explained with slides only** — no live demo needed. The audience gets the concept from the story and visuals. Trying to demo a toy transformer live adds complexity and undercuts the "scale matters" message.

### Phase Breakdown (Reduced)

| Phase | Objective | Deliverables |
|---|---|---|
| **1. Script** | Write the TED-style speaker script with timing marks | `keynote/SCRIPT.md` |
| **2. Core Code** | Single-file shapes engine + data | `shapes_core.py`, `shape_data.py` |
| **3. Demo App** | Streamlit with 4 tabs for the demo moments | `keynote/app.py` |
| **4. Slides** | ~30-slide Marp deck with Zara's journey | `keynote/slides/keynote.md` |
| **5. Rehearse** | Time it. Cut whatever makes it run over 20 min. | Speaker notes, timing log |

---

## Research That Backs It Up (If Someone Asks)

Keep this in your back pocket for Q&A — NOT in the talk itself.

| Topic | Key Reference | One-Liner |
|---|---|---|
| Image tokenization | Dosovitskiy et al., "An Image is Worth 16x16 Words" (2020) | ViT chops images into patches = tokens |
| Visual codebook | VQ-VAE (van den Oord, 2017), VQ-GAN (Esser, 2021) | Learned "standard cell library" of visual atoms |
| Joint embeddings | CLIP (Radford et al., OpenAI, 2021) | A photo and its caption become the same vector |
| Feature visualization | Chris Olah, Distill.pub (2017-2020) | CNNs learn edges → textures → shapes → objects |
| Attention in vision | ViT self-attention (2020) | Every image patch attends to every other patch |
| Image generation | DiT / Stable Diffusion 3 (Peebles & Xie, 2023) | Transformers generate images, not just text |
| Layout generation | LayoutTransformer (Gupta et al., 2021) | Layouts = sequences of [type, x, y, w, h] |
| Chip floorplanning | Mirhoseini et al. (Google, Nature 2021) | RL + ML for TPU macro placement |
| Analog layout AI | ALIGN (DARPA), MAGICAL (UT Austin) | Academic projects generating analog layouts from netlists |
| Commercial tools | Cadence Virtuoso, Synopsys Custom Compiler | ML-assisted constraint extraction and placement |

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Talk runs over 20 min | Rehearse with timer. Every section has a "cut" line — drop it and the story still works |
| Demo fails live | Slides include static screenshots of every demo. If Streamlit breaks, advance one slide — audience never knows |
| Audience is too technical / wants depth | "I have a full 6-workshop series that goes deep — QR code on the last slide" |
| Audience is too skeptical | The reveal section names real companies and Nature paper. This is not vaporware. |
| "Transformer" confusion | Call it "the generator" or "the brain" in the talk. Never say "transformer" without immediately clarifying "not the magnetic kind" |

---

## The Three Takeaways (Audience Leaves With)

1. **AI learns to see shapes the same way it learns to read text** — break it apart, map meaning, search, attend, generate, verify
2. **This is already happening in chip design** — Google, Cadence, Synopsys, academic projects
3. **AI is your copilot, not your replacement** — patterns are learnable; taste is not

---

## State Tracking

- **Current Phase:** Planning — refined for 20-min TED format
- **Plan Progress:** 0 of 5 phases
- **Last Action:** Plan rewritten for TED-style, 20-minute, light-hearted format
- **Next Action:** User approves → Phase 1 (Speaker Script)

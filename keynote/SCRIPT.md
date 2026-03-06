# SCRIPT: Zara Crash-Lands in Shape World

**20-Minute TED-Style Keynote**

## Act 1: The Hook & Tokenization (0:00 - 5:00)

**[Slide: Title: "An Alien Visits Shape World"]**

*(Walk to center stage, looking directly at the audience. No clicker yet.)*

I want you to imagine something impossible. Imagine you've never seen a shape. Not a circle. Not a square. Not a line. Nothing.

You come from a world of sound. You perceive the universe through vibrations, frequencies, echoes. You're brilliant — a scientist, in fact. But you have no concept of "edges" or "corners" or "color."

Now imagine you crash-land on Earth. You open your eyes for the first time. And you see... this.

**[Slide: Chaotic colorful shape collage]**

*(Pause. Let them look at the chaos.)*

Meet Zara. She's an alien. And she's about to go on the exact same journey that every AI goes on when we teach it to understand images.

Zara's first problem: she's overwhelmed. There's TOO MUCH to look at. Every pixel, every color, every edge — it's all noise. She needs to break what she sees into manageable pieces.

Her first attempt: look at every single pixel. One at a time. It's like trying to understand a song by listening to one air molecule vibrating at a time. Technically complete. Practically useless.

Her second attempt: memorize every complete shape. "That's a circle! That's a rectangle!" Works great — until she sees a shape she's never seen before. "What IS that?! It's not in my catalog!"

Her breakthrough: learn the PARTS. Corners repeat. Edges repeat. Curves repeat. She builds an alphabet of shape-parts — and suddenly she can describe ANY shape, even brand-new ones, as a combination of parts she already knows.

**[Slide: Visual showing raw image -> pixel grid -> shape catalog -> learned shape parts]**

By the way — this is exactly what happens when you upload a photo to ChatGPT. The image gets chopped into patches. Each patch becomes a token — a "shape-word." That's all tokenization is: turning something continuous into manageable pieces.

## Act 2: Embeddings & Vector Search (5:00 - 9:00)

Zara can identify shape-parts now. But she has a problem. She gave each part a number. Edge-type-42. Edge-type-43. But the numbers are meaningless — a sharp corner and a gentle curve are one number apart. She needs a way to capture that some shapes are SIMILAR and others are DIFFERENT.

So she builds a map.

**[Slide: Meaning Map demo - 2D scatter plot of shape embeddings]**

On this map, similar shapes live close together. Different shapes live far apart. And here's the magical part — the map captures RELATIONSHIPS. If she knows that a "triangle on top of a rectangle" means "house"... she can figure out that a "dome on top of a rectangle" might mean "mosque." The DIRECTION between shapes has meaning.

This is what AI researchers call embeddings. Every shape becomes a point in space. And "meaning" becomes... distance.

*(Beat — let that land)*

Now Zara has another problem. She's collected THOUSANDS of these shape arrangements. When she sees something new, she wants to say "Have I seen something like this before?" She needs a library — one where you don't search by name, you search by SIMILARITY.

You know that moment when you're browsing your phone photos and Google says "These look similar"? That's a vector database. It embeds your photo into numbers, then finds the nearest neighbors.

**[Slide: Shape Library demo - query shape with similar results]**

Raise your hand if you've ever flipped through old tape-outs looking for something "kind of like" what you're working on now.

*(Pause for laughs and hands)*

Congratulations — you're a human vector database.

## Act 3: Attention & Transformers (9:00 - 13:00)

Zara's getting good. She can identify shapes, know which ones are similar, and search her library. But she keeps making a mistake.

She sees a red circle. Fine — she knows circles. But she treats it the same whether it's on a traffic light or on a clown's nose. She doesn't get that the SAME shape means completely different things depending on what's AROUND it.

She needs to learn what humans do effortlessly: pay attention to context.

**[Slide: Red circle — two contexts (traffic light vs clown nose)]**

So Zara invents a spotlight system. Every shape gets to shine a spotlight on every other shape and ask: "Are you important to me?"

And here's the cool part — she doesn't use ONE spotlight. She uses four. One looks for shapes that are close together. One looks for shapes the same color. One looks for shapes that are aligned. One looks for shapes that contain each other.

**[Slide: Attention Heatmap demo - highlighting shape relationships]**

When she looks at a triangle sitting on top of a rectangle, the "alignment" spotlight goes crazy between them. When she looks at two matching triangles in a symmetric pattern, the "proximity and color" spotlights both fire.

This is attention — the breakthrough that transformed AI in 2017. It's how the machine learns that "it" in "The cat sat on the mat because IT was tired" refers to the cat, not the mat. And it's how vision AI learns that two shapes RELATE to each other across an image.

*(Transition beat)*

Now Zara has all the pieces. A way to see parts. A map of meaning. A library of patterns. Spotlights for context. She asks the big question:

Can I CREATE something new?

**[Slide: The Artist - Zara at a workbench]**

She stacks all her tools into one system. Feed in some shapes — the system predicts: what shape goes next, and WHERE should it go? One shape at a time, like writing a sentence word by word.

Her first attempt is... not great.

**[Slide: Hilariously bad shape arrangement - Bad AI House]**

*(Pause for laughs)*

But the architecture is right. It's the same architecture behind DALL-E, Stable Diffusion, and ChatGPT. The difference? Our little demo has ten thousand parameters. GPT-4 has over a trillion. Same engine — enormously more horsepower.

## Act 4: RAG (13:00 - 16:00)

Zara has one last problem. Sometimes her generator hallucinates. It creates shapes that look plausible but are nonsensical. A five-sided stop sign. A face with the mouth above the eyes.

Her solution is brilliantly humble: before creating something new, LOOK at what already exists. Search the library. Pull up similar examples. Use them as a guide. THEN generate.

**[Slide: RAG Pipeline - Search -> Retrieve -> Generate]**

This is called RAG — Retrieval-Augmented Generation. It's the reason ChatGPT can answer questions about today's news even though it was trained months ago. It looks things up first.

And it's the reason AI for YOUR world doesn't have to hallucinate layouts from scratch. It can retrieve proven designs from your own library — layouts that passed DRC, that taped out successfully — and suggest adaptations.

## The Reveal & Closing (16:00 - 20:00)

So let me recap Zara's journey.

**[Slide: Zara's complete journey - Six Icons]**

She learned to break images into pieces. She mapped meaning into space. She built a searchable library. She learned which relationships matter. She assembled a generator. And she learned to check her work against known examples.

Now... replace "shapes" with "polygons on metal layers." Replace "arrangements" with "layout floorplans." Replace "library" with "your corporate IP database."

**[Slide: Side-by-side: Zara's shapes vs Real Layout]**

The math is identical. The concepts are identical. The only difference is what the shapes represent.

Companies like Cadence, Synopsys, and Google are already applying these exact techniques. Google published a paper in Nature using AI for chip floorplanning. Cadence and Synopsys have ML-assisted placement in their latest tools. Research projects like ALIGN and MAGICAL are generating analog layouts from netlists.

It's not science fiction. It's happening.

I know what some of you are thinking. "Great. I'm going to be replaced by a robot."

You're not. And here's why.

Zara learned to see patterns. But she never learned TASTE. She doesn't know why you put those guard rings there. She doesn't feel the intuition that says "that parasitic is going to bite us." She doesn't have twenty years of tribal knowledge about what works at your foundry.

AI is the most talented intern you've ever had. It remembers every layout you've ever done. It can search by similarity in milliseconds. It can suggest a first-pass placement that's 80% there.

The last 20%? That's you. That's the art. And that's not going anywhere.

**[Slide: Thank You]**

Zara went from not knowing what a shape was... to being able to generate and verify arrangements. That's the GenAI journey, in twenty minutes.

Thank you.

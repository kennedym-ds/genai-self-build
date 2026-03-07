# In-Depth Repo Review: TED-Style Talk for Analog Layout Designers

**Date:** 2026-03-07  
**Scope:** Full repository review — keynote readiness for visual GenAI talk to ADI layout designers  
**Verdict:** STRONG foundation. Targeted improvements needed for audience alignment.

---

## Executive Summary

The repository is **technically excellent** — all 133 tests pass across 6 workshops, the narrative (Zara's journey) is compelling and complete, and the dual-stack architecture (HTML SPA + Python/Streamlit) gives you maximum delivery flexibility. The keynote script is well-written and already speaks directly to layout designers ("replace shapes with polygons on metal layers").

**Primary gap:** The interactive demos use generic geometric shapes rather than EDA-flavored primitives. For a room of analog layout engineers, bridging that last mile — naming things in their language — would transform "good talk" into "unforgettable talk."

---

## Health Check

| Area | Status | Details |
|------|--------|---------|
| Workshop 1 Tests | ✅ 19/19 | Character, word, BPE tokenizers |
| Workshop 2 Tests | ✅ 21/21 | Random, co-occurrence, prediction embeddings |
| Workshop 3 Tests | ✅ 22/22 | Flat, LSH, IVF search strategies |
| Workshop 4 Tests | ✅ 21/21 | Multi-head attention, causal masking |
| Workshop 5 Tests | ✅ 25/25 | Full transformer, generation |
| Workshop 6 Tests | ✅ 25/25 | RAG pipeline, retrieval + generation |
| **Total** | **✅ 133/133** | **Zero failures** |
| IDE Errors | ✅ 0 | Clean workspace |
| HTML SPA | ✅ Works | Zero dependencies, offline-ready |
| Keynote App | ✅ Works | Standalone Streamlit + HTML companion |
| Documentation | ✅ Complete | Story Guide, Teacher Guide, Under the Hood, User Guide |

---

## Strengths

### 1. Narrative Excellence
- Zara's alien journey is the perfect analogy for this audience — layout designers already think spatially
- The keynote script has a killer closing: "AI is the most talented intern you've ever had… The last 20%? That's you."
- Chapter transitions maintain tension and emotional arc across all 6 concepts
- The Feynman quote framing ("What I cannot create, I do not understand") resonates with engineering culture

### 2. Technical Depth Without Overwhelm
- Each workshop builds from scratch — no black-box imports
- Debug mode across all implementations lets curious engineers peek inside
- The "Under the Hood" doc bridges educational simplifications to production reality
- Test suites are comprehensive with emoji-rich output for demo visibility

### 3. Dual Delivery Flexibility
- **HTML SPA**: Zero-install, works offline, perfect for large conference rooms
- **Streamlit**: Interactive controls, better for smaller workshops
- **Keynote HTML app**: Stripped-down 4-tab version focused on live demo moments
- **Marp slides**: Professional dark-theme deck with speaker notes

### 4. Shape Engine Architecture
- `ShapeTokenizer`, `ShapeEmbedding`, `ShapeVectorDB`, `ShapeAttention` — clean, consistent APIs
- Visual path renders shapes as SVGs, making the "shapes = layout primitives" connection tangible
- Shared `TextUtils` prevents code duplication across modules

---

## Gaps & Recommendations

### HIGH PRIORITY — Audience Alignment

#### Gap 1: Generic Shapes → EDA Terminology
**Current:** Scenes use abstract names like "rectangle", "circle", "triangle"  
**Impact:** Layout designers won't instinctively connect to their daily work  
**Fix:** Add EDA-flavored aliases in `shape_data.js` and `keynote/shape_data.py`:
- "rectangle" → "Metal-1 trace" or "P-Well"
- "circle" → "Via" or "Contact"
- Scenes like "simple_house" → "Differential Pair Cell" or "Current Mirror Layout"
- Add 2-3 scenes that look like simplified layout cross-sections

#### Gap 2: Attention ↔ Parasitics Analogy
**Current:** Attention is explained as "spotlight of focus" (generic)  
**Impact:** Missing the most resonant analogy for this audience  
**Fix:** Add a presenter note or slide callout: *"Attention is like parasitic coupling — two parallel metal traces heavily 'attend' to each other (high capacitance), while a shielded route far away has low attention. The math is the same: proximity × interaction strength."*

#### Gap 3: Vector DB ↔ DRC/LVS Connection
**Current:** Vector DB is explained as "magic library shelves"  
**Impact:** Layout engineers search their IP libraries by similarity constantly  
**Fix:** Add a callout: *"You're already doing vector search when you flip through tape-outs looking for something 'kind of like' your current design. A vector DB just does it in milliseconds."* (Note: The keynote script already has this line — ensure the interactive demo reinforces it.)

#### Gap 4: Embeddings ↔ Matched Pair Placement
**Current:** "Similar words live close together on the map"  
**Impact:** Could be more visceral for this audience  
**Fix:** *"GenAI embeddings place words close together in space — just like you place matched transistors close together to share a thermal gradient. Proximity = similarity. The math is identical."*

### MEDIUM PRIORITY — Technical Polish

#### Gap 5: Keynote Slide Placeholder Images
**Current:** Marp deck references `https://via.placeholder.com/...` URLs  
**Impact:** Will show generic placeholder boxes during the live talk  
**Fix:** Replace with actual images in `keynote/slides/images/` — especially the "chaotic shape collage" and "Zara's journey" summary slides

#### Gap 6: `TestResult` Class Naming
**Current:** `class TestResult` in test files triggers `PytestCollectionWarning` if anyone runs `pytest`  
**Impact:** Minor — only affects developers who clone and test with pytest  
**Fix:** Add `__test__ = False` to the class, or rename to `TrackingResult`

#### Gap 7: Hover/Focus States in HTML SPA
**Current:** Interactive elements lack visual feedback for audience visibility  
**Impact:** Hard for back-row audience members to see what the presenter is clicking  
**Fix:** Add subtle glow/scale on hover for clickable elements in `css/index.css`

### LOW PRIORITY — Nice to Have

#### Gap 8: Global Test Runner
**Current:** Each workshop runs tests independently (`python test_*.py`)  
**Impact:** No single command validates everything  
**Fix:** Add a `run_all_tests.py` or PowerShell script

#### Gap 9: `shape_data.js` Scene Count
**Current:** Limited number of scenes for Vector DB "nearest neighbor" queries  
**Impact:** Demo may feel repetitive when searching for similar shapes  
**Fix:** Add 5-10 more scenes with varied complexity

---

## Keynote Delivery Checklist

- [ ] Replace placeholder images in `keynote/slides/keynote.md`
- [ ] Test `keynote/html_app/index.html` on the actual presentation laptop/projector
- [ ] Verify `streamlit run keynote/app.py` works in the presentation environment
- [ ] Pre-load scenes in the HTML app (avoid first-load jank)
- [ ] Practice the "raise your hand" moment (Act 2 — vector DB + searching tape-outs)
- [ ] Prepare 2-3 backup examples in case live demo hits edge case
- [ ] Consider adding one real layout screenshot for the "side-by-side: Zara's shapes vs Real Layout" slide

---

## State Tracking

- **Current Phase:** Review Complete
- **Plan Progress:** N/A (review, not implementation)
- **Last Action:** Full repository audit with test validation
- **Next Action:** User decides which improvements to prioritize for the talk

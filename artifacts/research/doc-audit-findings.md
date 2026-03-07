# Research: Documentation Audit for genai-self-build

**Date**: 2026-03-06
**Researcher**: researcher-agent
**Confidence**: High
**Tools Used**: read_file

## Summary
The `genai-self-build` repository has evolved from a pure Python/Streamlit workshop structure to a dual-interface system (HTML SPA + Python Workshops) and a Keynote demo. However, the documentation and slides have **not** been updated to reflect the `index.html` SPA or the `keynote/` presentation. All top-level documentation guides still erroneously treat the `app.py` Streamlit app as the one and only "Unified Demo", completely ignoring the JavaScript/HTML implementations. 

## Key Findings by Severity

### 🔴 SEVERITY 1: CRITICAL (Major Rewrites Needed)

#### `docs/USER_GUIDE.md`
1. **Accurate vs Outdated**: Python prerequisites and individual workshop guides are accurate for the *Python track*, but the guide completely misses the rest of the project.
2. **Specific Lines**: "Option 1: Unified Demo (Recommended for Overview)" strictly refers to `streamlit run app.py` (Line 60). Claims "No external libraries (pure Python)" which ignores the HTML/JS track.
3. **HTML SPA Mention**: ZERO mentions of `index.html`, `js/`, or the dual-interface setup.
4. **Action**: **MAJOR REWRITE**. Must introduce the Web UI vs Python Workspaces distinction early on.

#### `docs/TEACHER_GUIDE.md`
1. **Accurate vs Outdated**: The conceptual pacing, analogies, and pedagogy are pristine. The technical execution instructions are dangerously outdated.
2. **Specific Lines**: "Backup Plans: Google Colab notebooks" (Line 294), "Run unified demo: `streamlit run app.py`" (Line 310). Doesn't give instructors any directions on how to use the primary `index.html` or Keynote demos.
3. **HTML SPA Mention**: ZERO mentions.
4. **Action**: **MAJOR REWRITE**. Needs to guide teachers on using the SPA for presentation and Python track for hands-on, plus add the `keynote/` instructions.

#### `docs/workshop-plan.md`
1. **Accurate vs Outdated**: Timeframes and topics are accurate. Architecture maps and file trees are highly inaccurate.
2. **Specific Lines**: "Repository Structure" (Lines 371-395) entirely omits `index.html`, `js/`, `css/`, and `keynote/`. "Tech Stack" under every workshop specifies Python 3.10+ and no external libraries.
3. **HTML SPA Mention**: ZERO mentions.
4. **Action**: **MAJOR REWRITE**. Update the repo tree and technology sections to note the side-by-side JS/HTML and Python implementations.

#### `.github/copilot-instructions.md`
1. **Accurate vs Outdated**: The alien analogies and themes are accurate. The repository rules are broken.
2. **Specific Lines**: "Workshop Structure" (Lines 11-20) specifies `app.py` as the "Unified Demo" and misses the new frontends. "Streamlit App Pattern" (Lines 80+) forces agents to build Streamlit apps, neglecting the new JS pipeline.
3. **HTML SPA Mention**: ZERO mentions.
4. **Action**: **MAJOR REWRITE**. Must redefine repository structure and specify rules for HTML/JS styling alongside Streamlit rules.

---

### 🟠 SEVERITY 2: MODERATE (Minor Updates Needed)

#### `docs/UNDER_THE_HOOD.md`
1. **Accurate vs Outdated**: The algorithm theory and Python debug mechanics are completely accurate. Outdated in scope.
2. **Specific Lines**: Focuses 100% on `tokenizer.py` and Python implementations.
3. **HTML SPA Mention**: ZERO mentions.
4. **Action**: **MINOR UPDATE**. Rename or add a disclaimer that this guide specifically applies to the Python Workshop Implementations, or expand it to include the `js/ml/` logic.

#### `docs/STORY_GUIDE.md`
1. **Accurate vs Outdated**: The Zara narrative arc maps 1:1 with both the Python and new SPA tracks perfectly.
2. **Specific Lines**: "Transitions: Use the scripts from the Story Guide to bridge workshops" (Line 196) implies a manual live lecture, not acknowledging that the `index.html` chapter navigation now beautifully handles these beats interactively.
3. **HTML SPA Mention**: ZERO mentions.
4. **Action**: **MINOR UPDATE**. Call out the UI presentation of Zara's journey in the HTML app. 

#### `workshops/01-tokenization/README.md` (and other workshop READMEs)
1. **Accurate vs Outdated**: Accurate instructions for the Python scripts.
2. **Specific Lines**: Points users exclusively to `streamlit run app.py`. 
3. **HTML SPA Mention**: ZERO mentions.
4. **Action**: **MINOR UPDATE**. Note the existence of the browser-based visualization for those who just want the concepts without setup.

---

### 🟡 SEVERITY 3: LOW (Fine As-Is / Archival)

#### Slide Decks (`workshops/*/slides/slides.md`)
1. **Accurate vs Outdated**: Mostly accurate. They teach concepts logically.
2. **Specific Lines**: E.g. `01-tokenization/slides/slides.md` references the `tokenizer.py` file directly for the live demo part. 
3. **HTML SPA Mention**: ZERO mentions.
4. **Action**: **FINE AS-IS**. If the slides are to accompany the Python workshops, they are accurate. If the Main Demo is now HTML/Keynote, the slides might just be collateral options.

#### Cheatsheets & Q&As (`workshops/*/cheatsheet.md`, `qna.md`)
1. **Accurate vs Outdated**: Conceptual facts are timeless.
2. **Specific Lines**: None out of place.
3. **HTML SPA Mention**: ZERO mentions (N/A).
4. **Action**: **FINE AS-IS**.

#### `REVIEW_SUMMARY.md`
1. **Accurate vs Outdated**: Completely out of date. It tracks a historical set of improvements (the "debug mode" addition). 
2. **Specific Lines**: "Recommended Next Steps" are now obsolete.
3. **HTML SPA Mention**: ZERO mentions.
4. **Action**: **DEPRECATE / ARCHIVE**. Contains no forward-looking value.

---

## Validating General Inquiries
- **`app.py` at Root**: The file still exists and **still functions correctly** as a Python-centric unified wrapper around `workshops/01-06/`. The file paths it uses are valid. However, the docs calling it the *only/primary* conference demo is what's wrong.
- **Bad File Paths**: `workshop-plan.md` outlines a directory tree that is fundamentally missing half the codebase. 
- **Zara Narrative**: The Zara storyline is consistent. The narrative arc in `STORY_GUIDE.md` aligns with the UI paths you mentioned in the HTML wrapper. It just needs to formally mention that `index.html` carries this narrative.

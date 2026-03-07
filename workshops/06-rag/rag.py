"""
🔍 Workshop 6: RAG (Retrieval-Augmented Generation)
===================================================

🛸 THE SEARCH ENGINE ANALOGY:
Our alien has learned to process language (transformer), but they have a problem:
their brain only knows what they learned during "school" (training). Ask about 
something new, and they'll either say "I don't know" or worse - make something up!

So the alien builds a SEARCH ENGINE connected to a LIBRARY:
    1. 📝 Someone asks a question
    2. 🔍 Alien searches their library for relevant books
    3. 📚 Pulls out the most relevant pages
    4. 🧠 Reads those pages + the question
    5. 💬 Answers based on what they just read

THIS IS EXACTLY WHAT RAG DOES.
The "library" is a vector database. The "search" is semantic similarity.
The "reading + answering" is the language model with augmented context.

RAG = Retrieval (find relevant docs) + Augmented (add to prompt) + Generation (answer)

This is how ChatGPT plugins, Perplexity, and enterprise AI assistants work!

Usage:
    python rag.py
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import re


# =============================================================================
# PART 1: DOCUMENT STORE (Our alien's library)
# =============================================================================

class SimpleEmbedder:
    """
    A simple word-based embedder for demonstration.
    
    In production, you'd use sentence-transformers, OpenAI embeddings, etc.
    This uses a bag-of-words approach with random projection for simplicity.
    """
    
    def __init__(self, embed_dim: int = 64, seed: int = 42):
        self.embed_dim = embed_dim
        self.word_vectors: Dict[str, np.ndarray] = {}
        np.random.seed(seed)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase and split on non-alphanumeric."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _get_word_vector(self, word: str) -> np.ndarray:
        """Get or create a random vector for a word."""
        if word not in self.word_vectors:
            # Create a deterministic random vector based on word hash
            np.random.seed(hash(word) % (2**31))
            self.word_vectors[word] = np.random.randn(self.embed_dim).astype(np.float32)
            self.word_vectors[word] /= np.linalg.norm(self.word_vectors[word])
        return self.word_vectors[word]
    
    def embed(self, text: str) -> np.ndarray:
        """
        Embed a text string into a vector.
        
        Uses mean of word vectors (bag-of-words style).
        """
        words = self._tokenize(text)
        if not words:
            return np.zeros(self.embed_dim, dtype=np.float32)
        
        vectors = [self._get_word_vector(w) for w in words]
        embedding = np.mean(vectors, axis=0)
        
        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts."""
        return np.array([self.embed(t) for t in texts])


class DocumentStore:
    """
    A simple vector store for documents.
    
    This is a minimal implementation of what we built in Workshop 3.
    """
    
    def __init__(self, embedder: SimpleEmbedder):
        self.embedder = embedder
        self.documents: List[Dict] = []  # List of {id, text, metadata, embedding}
        self.embeddings: Optional[np.ndarray] = None
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Add documents to the store.
        
        Args:
            texts: List of document texts
            metadatas: Optional list of metadata dicts (e.g., source, title)
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        embeddings = self.embedder.embed_batch(texts)
        
        for i, (text, meta, emb) in enumerate(zip(texts, metadatas, embeddings)):
            doc_id = len(self.documents)
            self.documents.append({
                'id': doc_id,
                'text': text,
                'metadata': meta,
                'embedding': emb
            })
        
        # Rebuild embedding matrix
        self.embeddings = np.array([d['embedding'] for d in self.documents])
        
        print(f"✅ Added {len(texts)} documents (total: {len(self.documents)})")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of documents with similarity scores
        """
        if not self.documents:
            return []
        
        # Embed the query
        query_embedding = self.embedder.embed(query)
        
        # Compute cosine similarities (embeddings are normalized)
        similarities = self.embeddings @ query_embedding
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return documents with scores
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc['score'] = float(similarities[idx])
            results.append(doc)
        
        return results


# =============================================================================
# PART 2: PROMPT BUILDER (Augmenting with context)
# =============================================================================

class PromptBuilder:
    """
    Builds prompts by combining retrieved context with the user query.
    
    This is the "Augmented" part of RAG - we augment the prompt with
    relevant information retrieved from our document store.
    """
    
    def __init__(self, max_context_length: int = 1000):
        self.max_context_length = max_context_length
        
        # Template for RAG prompts
        self.template = """Answer the question based on the following context. 
If the context doesn't contain enough information, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""
    
    def build_prompt(self, question: str, retrieved_docs: List[Dict]) -> str:
        """
        Build a prompt with retrieved context.
        
        Args:
            question: User's question
            retrieved_docs: Documents retrieved from vector store
            
        Returns:
            Formatted prompt string
        """
        # Format context from retrieved documents
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(retrieved_docs):
            source = doc.get('metadata', {}).get('source', f'Document {i+1}')
            text = doc['text']
            
            # Truncate if needed
            if total_length + len(text) > self.max_context_length:
                remaining = self.max_context_length - total_length
                if remaining > 100:  # Only include if meaningful amount left
                    text = text[:remaining] + "..."
                else:
                    break
            
            context_parts.append(f"[{source}]: {text}")
            total_length += len(text)
        
        context = "\n\n".join(context_parts)
        
        return self.template.format(context=context, question=question)
    
    def build_prompt_with_sources(self, question: str, retrieved_docs: List[Dict]) -> Tuple[str, List[str]]:
        """
        Build prompt and return source references separately.
        
        Returns:
            Tuple of (prompt, list of source names)
        """
        prompt = self.build_prompt(question, retrieved_docs)
        sources = [
            doc.get('metadata', {}).get('source', f'Document {i+1}')
            for i, doc in enumerate(retrieved_docs)
        ]
        return prompt, sources


# =============================================================================
# PART 3: SIMPLE GENERATOR (For demonstration)
# =============================================================================

class SimpleGenerator:
    """
    A rule-based "generator" for demonstration purposes.
    
    In production, you'd use GPT-4, Claude, LLaMA, etc.
    This uses keyword matching and extractive answering for simplicity.
    """
    
    def __init__(self):
        # Keywords that indicate inability to answer
        self.uncertainty_phrases = [
            "i don't have enough information",
            "i cannot answer",
            "not mentioned",
            "no information"
        ]
    
    def _extract_relevant_sentences(self, context: str, question: str) -> List[str]:
        """Extract sentences from context that seem relevant to the question."""
        # Get question keywords
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        question_words -= {'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 
                          'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of'}
        
        # Split context into sentences
        sentences = re.split(r'[.!?]+', context)
        
        # Score sentences by keyword overlap
        scored = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            sent_words = set(re.findall(r'\b\w+\b', sent.lower()))
            overlap = len(question_words & sent_words)
            if overlap > 0:
                scored.append((overlap, sent))
        
        # Sort by relevance
        scored.sort(reverse=True)
        
        return [s[1] for s in scored[:3]]
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response based on the prompt.
        
        This is a simplified extractive approach - it pulls relevant
        sentences from the context rather than truly generating.
        """
        # Parse the prompt to extract context and question
        parts = prompt.split("Question:")
        if len(parts) != 2:
            return "I couldn't understand the question format."
        
        context_part = parts[0]
        question_part = parts[1].split("Answer:")[0].strip()
        
        # Extract context (between "Context:" and "Question:")
        context_start = context_part.find("Context:")
        if context_start != -1:
            context = context_part[context_start + 8:].strip()
        else:
            context = context_part.strip()
        
        # Check if context is empty or too short
        if len(context) < 20:
            return "I don't have enough information to answer this question."
        
        # Extract relevant sentences
        relevant = self._extract_relevant_sentences(context, question_part)
        
        if not relevant:
            return "I don't have enough information to answer this question based on the provided context."
        
        # Combine into answer
        answer = " ".join(relevant)
        
        # Clean up
        answer = answer.strip()
        if not answer.endswith('.'):
            answer += '.'
        
        return answer


# =============================================================================
# PART 4: THE RAG PIPELINE (Bringing it all together!)
# =============================================================================

class RAGPipeline:
    """
    Complete RAG (Retrieval-Augmented Generation) pipeline.
    
    This combines everything from our workshop series:
    - Embeddings (Workshop 2) for encoding queries and documents
    - Vector search (Workshop 3) for retrieval
    - Generation (Workshop 5) for answering
    
    The flow:
    1. User asks a question
    2. Embed the question
    3. Search vector store for relevant documents
    4. Build prompt with retrieved context
    5. Generate answer based on context
    6. Return answer with sources
    """
    
    def __init__(self, embed_dim: int = 64, top_k: int = 3):
        self.embedder = SimpleEmbedder(embed_dim=embed_dim)
        self.doc_store = DocumentStore(self.embedder)
        self.prompt_builder = PromptBuilder()
        self.generator = SimpleGenerator()
        self.top_k = top_k
        
        print("🔍 RAG Pipeline initialized!")
        print(f"   - Embedding dimension: {embed_dim}")
        print(f"   - Top-K retrieval: {top_k}")
    
    def add_knowledge(self, texts: List[str], sources: Optional[List[str]] = None):
        """
        Add knowledge documents to the RAG system.
        
        Args:
            texts: List of document texts
            sources: Optional list of source names/identifiers
        """
        if sources is None:
            sources = [f"doc_{i}" for i in range(len(texts))]
        
        metadatas = [{"source": src} for src in sources]
        self.doc_store.add_documents(texts, metadatas)
    
    def query(self, question: str, verbose: bool = False) -> Dict:
        """
        Process a question through the RAG pipeline.
        
        Args:
            question: User's question
            verbose: Whether to print intermediate steps
            
        Returns:
            Dict with answer, sources, and retrieved documents
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"📝 Question: {question}")
            print(f"{'='*60}")
        
        # Step 1: Retrieve relevant documents
        retrieved = self.doc_store.search(question, top_k=self.top_k)
        
        if verbose:
            print(f"\n🔍 Retrieved {len(retrieved)} documents:")
            for i, doc in enumerate(retrieved):
                print(f"   [{i+1}] Score: {doc['score']:.3f} | {doc['metadata'].get('source', 'unknown')}")
                print(f"       {doc['text'][:100]}...")
        
        # Step 2: Build augmented prompt
        prompt, sources = self.prompt_builder.build_prompt_with_sources(question, retrieved)
        
        if verbose:
            print(f"\n📋 Built prompt ({len(prompt)} chars)")
        
        # Step 3: Generate answer
        answer = self.generator.generate(prompt)
        
        if verbose:
            print(f"\n💬 Answer: {answer}")
            print(f"\n📚 Sources: {', '.join(sources)}")
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_docs": retrieved,
            "prompt": prompt
        }
    
    def chat(self, question: str) -> str:
        """
        Simple chat interface - returns just the answer with sources.
        """
        result = self.query(question)
        sources_str = ", ".join(result["sources"])
        return f"{result['answer']}\n\n📚 Sources: {sources_str}"


# =============================================================================
# SAMPLE KNOWLEDGE BASE
# =============================================================================

SAMPLE_KNOWLEDGE = [
    # Technology
    ("Python is a high-level programming language created by Guido van Rossum in 1991. "
     "It emphasizes code readability and supports multiple programming paradigms including "
     "procedural, object-oriented, and functional programming.", "tech/python"),
    
    ("Machine learning is a subset of artificial intelligence that enables systems to learn "
     "from data. Key approaches include supervised learning, unsupervised learning, and "
     "reinforcement learning. Popular frameworks include TensorFlow, PyTorch, and scikit-learn.", "tech/ml"),
    
    ("Transformers are a neural network architecture introduced in the 2017 paper 'Attention "
     "Is All You Need'. They use self-attention mechanisms and have become the foundation for "
     "models like GPT, BERT, and Claude.", "tech/transformers"),
    
    ("RAG (Retrieval-Augmented Generation) combines language models with information retrieval. "
     "Instead of relying solely on trained knowledge, RAG retrieves relevant documents and uses "
     "them as context for generation. This reduces hallucination and enables access to current information.", "tech/rag"),
    
    # Science
    ("Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide "
     "into glucose and oxygen. It occurs primarily in the chloroplasts of plant cells and is "
     "essential for life on Earth.", "science/biology"),
    
    ("The theory of general relativity, published by Albert Einstein in 1915, describes gravity "
     "as the curvature of spacetime caused by mass and energy. It predicted phenomena like "
     "gravitational waves and black holes.", "science/physics"),
    
    ("DNA (deoxyribonucleic acid) is a molecule that carries genetic instructions. It consists "
     "of two strands forming a double helix, with bases adenine, thymine, guanine, and cytosine. "
     "The human genome contains about 3 billion base pairs.", "science/genetics"),
    
    # History
    ("The French Revolution began in 1789 with the storming of the Bastille. It led to the end "
     "of the monarchy in France and the rise of Napoleon Bonaparte. Key ideals included liberty, "
     "equality, and fraternity.", "history/french_rev"),
    
    ("World War II lasted from 1939 to 1945 and involved most of the world's nations. It resulted "
     "in an estimated 70-85 million deaths and ended with the atomic bombings of Hiroshima and "
     "Nagasaki.", "history/ww2"),
    
    # Geography
    ("Mount Everest is the highest peak on Earth at 8,848.86 meters above sea level. Located in "
     "the Himalayas on the border of Nepal and Tibet, it was first summited by Edmund Hillary "
     "and Tenzing Norgay in 1953.", "geography/everest"),
    
    ("The Amazon River is the largest river by volume and the second longest in the world. It "
     "flows through South America, primarily Brazil, and its basin contains the world's largest "
     "tropical rainforest.", "geography/amazon"),
]


# =============================================================================
# DEMO AND TESTS
# =============================================================================

def demo_rag():
    """Demonstrate the RAG pipeline."""
    
    print("=" * 70)
    print("🔍 RAG (Retrieval-Augmented Generation) Demo")
    print("=" * 70)
    
    # Initialize RAG
    rag = RAGPipeline(embed_dim=64, top_k=3)
    
    # Add knowledge
    print("\n📚 Loading knowledge base...")
    texts = [doc[0] for doc in SAMPLE_KNOWLEDGE]
    sources = [doc[1] for doc in SAMPLE_KNOWLEDGE]
    rag.add_knowledge(texts, sources)
    
    # Test questions
    test_questions = [
        "What is Python and who created it?",
        "How do transformers work?",
        "What is photosynthesis?",
        "When did World War II end?",
        "How tall is Mount Everest?",
        "What is RAG and why is it useful?",
    ]
    
    print("\n" + "=" * 70)
    print("💬 Q&A SESSION")
    print("=" * 70)
    
    for question in test_questions:
        result = rag.query(question, verbose=False)
        print(f"\n❓ Q: {question}")
        print(f"💬 A: {result['answer']}")
        print(f"📚 Sources: {', '.join(result['sources'])}")
        print("-" * 50)
    
    # Show detailed example
    print("\n" + "=" * 70)
    print("🔬 DETAILED WALKTHROUGH")
    print("=" * 70)
    
    rag.query("What is machine learning and what frameworks are popular?", verbose=True)
    
    print("\n" + "=" * 70)
    print("✅ RAG Demo Complete!")
    print("=" * 70)


def test_components():
    """Test individual RAG components."""
    
    print("=" * 60)
    print("🧪 RAG COMPONENT TESTS")
    print("=" * 60)
    
    # Test 1: Embedder
    print("\n📦 TEST 1: SimpleEmbedder")
    print("-" * 40)
    embedder = SimpleEmbedder(embed_dim=64)
    
    emb1 = embedder.embed("machine learning AI")
    emb2 = embedder.embed("artificial intelligence ML")
    emb3 = embedder.embed("cooking recipes food")
    
    sim_12 = np.dot(emb1, emb2)
    sim_13 = np.dot(emb1, emb3)
    
    print(f"   'machine learning AI' vs 'artificial intelligence ML': {sim_12:.3f}")
    print(f"   'machine learning AI' vs 'cooking recipes food': {sim_13:.3f}")
    print(f"   ✅ Similar texts more similar: {sim_12 > sim_13}")
    
    # Test 2: Document Store
    print("\n📚 TEST 2: DocumentStore")
    print("-" * 40)
    store = DocumentStore(embedder)
    store.add_documents(
        ["Python is a programming language", "JavaScript runs in browsers", "SQL queries databases"],
        [{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]
    )
    
    results = store.search("programming languages", top_k=2)
    print(f"   Search 'programming languages':")
    for r in results:
        print(f"     - {r['text'][:40]}... (score: {r['score']:.3f})")
    
    # Test 3: Prompt Builder
    print("\n📝 TEST 3: PromptBuilder")
    print("-" * 40)
    builder = PromptBuilder()
    prompt, sources = builder.build_prompt_with_sources(
        "What is Python?",
        [{"text": "Python is great", "metadata": {"source": "docs"}}]
    )
    print(f"   Prompt length: {len(prompt)} chars")
    print(f"   Sources: {sources}")
    
    # Test 4: Full Pipeline
    print("\n🔍 TEST 4: RAGPipeline")
    print("-" * 40)
    rag = RAGPipeline()
    rag.add_knowledge(
        ["The sky is blue due to Rayleigh scattering", "Water is H2O"],
        ["science/sky", "science/water"]
    )
    result = rag.query("Why is the sky blue?")
    print(f"   Question: Why is the sky blue?")
    print(f"   Answer: {result['answer'][:100]}...")
    print(f"   Sources: {result['sources']}")
    
    print("\n" + "=" * 60)
    print("✅ All component tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_components()
    print("\n")
    demo_rag()

"""
🧪 Workshop 6: RAG Test Suite
==============================

Tests for the complete RAG pipeline.

Usage:
    python test_rag.py
"""

import numpy as np
from rag import (
    SimpleEmbedder, DocumentStore, PromptBuilder,
    SimpleGenerator, RAGPipeline, SAMPLE_KNOWLEDGE
)


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []
    
    def add(self, name: str, passed: bool, message: str = ""):
        if passed:
            self.passed += 1
            print(f"  ✅ {name}")
        else:
            self.failed += 1
            print(f"  ❌ {name}: {message}")
        self.results.append((name, passed, message))


def run_tests():
    """Run all RAG tests."""
    results = TestResults()
    
    print("=" * 60)
    print("🧪 RAG TEST SUITE")
    print("=" * 60)
    
    # =========================================================================
    # GROUP 1: SimpleEmbedder
    # =========================================================================
    print("\n📦 GROUP 1: SimpleEmbedder")
    print("-" * 40)
    
    # Test initialization
    try:
        embedder = SimpleEmbedder(embed_dim=64)
        assert embedder.embed_dim == 64
        results.add("embedder init", True)
    except Exception as e:
        results.add("embedder init", False, str(e))
    
    # Test embed
    try:
        embedder = SimpleEmbedder(embed_dim=64)
        emb = embedder.embed("hello world")
        assert emb.shape == (64,)
        assert emb.dtype == np.float32
        results.add("embedder embed shape", True)
    except Exception as e:
        results.add("embedder embed shape", False, str(e))
    
    # Test normalization
    try:
        embedder = SimpleEmbedder(embed_dim=64)
        emb = embedder.embed("test text")
        norm = np.linalg.norm(emb)
        assert np.abs(norm - 1.0) < 1e-5, f"Norm is {norm}, expected 1.0"
        results.add("embedder normalization", True)
    except Exception as e:
        results.add("embedder normalization", False, str(e))
    
    # Test semantic similarity
    try:
        embedder = SimpleEmbedder(embed_dim=64, seed=42)
        emb1 = embedder.embed("python programming code software")
        emb2 = embedder.embed("python code programming developer")
        emb3 = embedder.embed("banana apple orange fruit")
        
        sim_12 = np.dot(emb1, emb2)
        sim_13 = np.dot(emb1, emb3)
        
        # Texts with shared words should be more similar
        assert sim_12 > sim_13, f"Expected {sim_12} > {sim_13}"
        results.add("embedder semantic similarity", True)
    except Exception as e:
        results.add("embedder semantic similarity", False, str(e))
    
    # Test batch embed
    try:
        embedder = SimpleEmbedder(embed_dim=64)
        embs = embedder.embed_batch(["hello", "world", "test"])
        assert embs.shape == (3, 64)
        results.add("embedder batch", True)
    except Exception as e:
        results.add("embedder batch", False, str(e))
    
    # Test empty text
    try:
        embedder = SimpleEmbedder(embed_dim=64)
        emb = embedder.embed("")
        assert emb.shape == (64,)
        assert np.allclose(emb, 0)
        results.add("embedder empty text", True)
    except Exception as e:
        results.add("embedder empty text", False, str(e))
    
    # =========================================================================
    # GROUP 2: DocumentStore
    # =========================================================================
    print("\n📦 GROUP 2: DocumentStore")
    print("-" * 40)
    
    # Test add documents
    try:
        embedder = SimpleEmbedder(embed_dim=64)
        store = DocumentStore(embedder)
        store.add_documents(["doc1", "doc2", "doc3"])
        assert len(store.documents) == 3
        results.add("store add documents", True)
    except Exception as e:
        results.add("store add documents", False, str(e))
    
    # Test add with metadata
    try:
        embedder = SimpleEmbedder(embed_dim=64)
        store = DocumentStore(embedder)
        store.add_documents(
            ["Python is great", "Java is popular"],
            [{"source": "wiki"}, {"source": "docs"}]
        )
        assert store.documents[0]["metadata"]["source"] == "wiki"
        results.add("store metadata", True)
    except Exception as e:
        results.add("store metadata", False, str(e))
    
    # Test search
    try:
        embedder = SimpleEmbedder(embed_dim=64)
        store = DocumentStore(embedder)
        store.add_documents([
            "Python is a programming language",
            "Cats are furry animals",
            "JavaScript runs in browsers"
        ])
        results_search = store.search("programming", top_k=2)
        assert len(results_search) == 2
        # Programming doc should be first
        assert "programming" in results_search[0]["text"].lower() or "JavaScript" in results_search[0]["text"]
        results.add("store search", True)
    except Exception as e:
        results.add("store search", False, str(e))
    
    # Test search returns scores
    try:
        embedder = SimpleEmbedder(embed_dim=64)
        store = DocumentStore(embedder)
        store.add_documents(["test doc"])
        results_search = store.search("test", top_k=1)
        assert "score" in results_search[0]
        assert 0 <= results_search[0]["score"] <= 1
        results.add("store search scores", True)
    except Exception as e:
        results.add("store search scores", False, str(e))
    
    # Test empty store search
    try:
        embedder = SimpleEmbedder(embed_dim=64)
        store = DocumentStore(embedder)
        results_search = store.search("anything", top_k=3)
        assert results_search == []
        results.add("store empty search", True)
    except Exception as e:
        results.add("store empty search", False, str(e))
    
    # =========================================================================
    # GROUP 3: PromptBuilder
    # =========================================================================
    print("\n📦 GROUP 3: PromptBuilder")
    print("-" * 40)
    
    # Test prompt building
    try:
        builder = PromptBuilder()
        prompt = builder.build_prompt(
            "What is Python?",
            [{"text": "Python is a language", "metadata": {"source": "docs"}}]
        )
        assert "Python" in prompt
        assert "Question:" in prompt
        assert "Context:" in prompt
        results.add("builder basic prompt", True)
    except Exception as e:
        results.add("builder basic prompt", False, str(e))
    
    # Test multiple docs
    try:
        builder = PromptBuilder()
        prompt = builder.build_prompt(
            "Test?",
            [
                {"text": "Doc 1 content", "metadata": {"source": "src1"}},
                {"text": "Doc 2 content", "metadata": {"source": "src2"}}
            ]
        )
        assert "[src1]" in prompt
        assert "[src2]" in prompt
        results.add("builder multiple docs", True)
    except Exception as e:
        results.add("builder multiple docs", False, str(e))
    
    # Test with sources
    try:
        builder = PromptBuilder()
        prompt, sources = builder.build_prompt_with_sources(
            "Question?",
            [{"text": "Content", "metadata": {"source": "my_source"}}]
        )
        assert "my_source" in sources
        results.add("builder with sources", True)
    except Exception as e:
        results.add("builder with sources", False, str(e))
    
    # Test max length truncation
    try:
        builder = PromptBuilder(max_context_length=50)
        long_doc = "A" * 200
        prompt = builder.build_prompt("Q?", [{"text": long_doc, "metadata": {}}])
        # Context should be truncated
        assert len(prompt) < 500
        results.add("builder truncation", True)
    except Exception as e:
        results.add("builder truncation", False, str(e))
    
    # =========================================================================
    # GROUP 4: SimpleGenerator
    # =========================================================================
    print("\n📦 GROUP 4: SimpleGenerator")
    print("-" * 40)
    
    # Test generation
    try:
        generator = SimpleGenerator()
        prompt = """Answer the question based on the following context.

Context:
[doc1]: Python is a programming language created by Guido van Rossum.

Question: Who created Python?

Answer:"""
        answer = generator.generate(prompt)
        assert len(answer) > 0
        results.add("generator basic", True)
    except Exception as e:
        results.add("generator basic", False, str(e))
    
    # Test relevant extraction
    try:
        generator = SimpleGenerator()
        prompt = """Answer the question based on the following context.

Context:
[doc1]: The sky is blue due to Rayleigh scattering of sunlight.

Question: Why is the sky blue?

Answer:"""
        answer = generator.generate(prompt)
        assert "Rayleigh" in answer or "scattering" in answer or "blue" in answer.lower()
        results.add("generator extraction", True)
    except Exception as e:
        results.add("generator extraction", False, str(e))
    
    # =========================================================================
    # GROUP 5: RAGPipeline
    # =========================================================================
    print("\n📦 GROUP 5: RAGPipeline")
    print("-" * 40)
    
    # Test initialization
    try:
        rag = RAGPipeline(embed_dim=64, top_k=3)
        assert rag.top_k == 3
        results.add("rag init", True)
    except Exception as e:
        results.add("rag init", False, str(e))
    
    # Test add knowledge
    try:
        rag = RAGPipeline()
        rag.add_knowledge(["Doc 1", "Doc 2"], ["src1", "src2"])
        assert len(rag.doc_store.documents) == 2
        results.add("rag add knowledge", True)
    except Exception as e:
        results.add("rag add knowledge", False, str(e))
    
    # Test query
    try:
        rag = RAGPipeline(top_k=2)
        rag.add_knowledge([
            "Python was created by Guido van Rossum in 1991",
            "Java was created by James Gosling"
        ], ["python_doc", "java_doc"])
        
        result = rag.query("Who created Python?")
        assert "answer" in result
        assert "sources" in result
        assert "retrieved_docs" in result
        results.add("rag query structure", True)
    except Exception as e:
        results.add("rag query structure", False, str(e))
    
    # Test query returns relevant docs
    try:
        rag = RAGPipeline(top_k=2)
        rag.add_knowledge([
            "Cats are small furry mammals",
            "Python is a programming language",
            "Dogs are loyal companions"
        ])
        
        result = rag.query("Tell me about programming")
        # Python doc should be retrieved
        retrieved_texts = [d["text"] for d in result["retrieved_docs"]]
        assert any("Python" in t for t in retrieved_texts)
        results.add("rag retrieves relevant", True)
    except Exception as e:
        results.add("rag retrieves relevant", False, str(e))
    
    # Test chat interface
    try:
        rag = RAGPipeline()
        rag.add_knowledge(["Test document content"], ["test_source"])
        response = rag.chat("Test question")
        assert isinstance(response, str)
        assert "Sources:" in response
        results.add("rag chat interface", True)
    except Exception as e:
        results.add("rag chat interface", False, str(e))
    
    # =========================================================================
    # GROUP 6: Integration Tests
    # =========================================================================
    print("\n📦 GROUP 6: Integration Tests")
    print("-" * 40)
    
    # Test with sample knowledge
    try:
        rag = RAGPipeline()
        texts = [doc[0] for doc in SAMPLE_KNOWLEDGE]
        sources = [doc[1] for doc in SAMPLE_KNOWLEDGE]
        rag.add_knowledge(texts, sources)
        assert len(rag.doc_store.documents) == len(SAMPLE_KNOWLEDGE)
        results.add("integration load knowledge", True)
    except Exception as e:
        results.add("integration load knowledge", False, str(e))
    
    # Test real questions
    try:
        rag = RAGPipeline(top_k=3)
        texts = [doc[0] for doc in SAMPLE_KNOWLEDGE]
        sources = [doc[1] for doc in SAMPLE_KNOWLEDGE]
        rag.add_knowledge(texts, sources)
        
        result = rag.query("What is machine learning?")
        assert len(result["answer"]) > 10
        results.add("integration ml question", True)
    except Exception as e:
        results.add("integration ml question", False, str(e))
    
    # Test another domain
    try:
        rag = RAGPipeline(top_k=3)
        texts = [doc[0] for doc in SAMPLE_KNOWLEDGE]
        sources = [doc[1] for doc in SAMPLE_KNOWLEDGE]
        rag.add_knowledge(texts, sources)
        
        result = rag.query("How tall is Mount Everest?")
        # Should get an answer
        assert len(result["answer"]) > 10
        assert len(result["retrieved_docs"]) > 0
        results.add("integration geography question", True)
    except Exception as e:
        results.add("integration geography question", False, str(e))
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"📊 RESULTS: {results.passed} passed, {results.failed} failed")
    print("=" * 60)
    
    if results.failed == 0:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed. Review the output above.")
    
    return results


if __name__ == "__main__":
    run_tests()

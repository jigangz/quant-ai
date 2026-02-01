"""
RAG Answer Service

Retrieves relevant documents and generates a summary answer.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.rag.index import get_rag_index

logger = logging.getLogger(__name__)


class Evidence(BaseModel):
    """A piece of evidence retrieved by RAG."""
    
    id: str
    type: str
    text: str
    score: float
    metadata: dict[str, Any] = {}


class RAGAnswer(BaseModel):
    """Response from RAG answering."""
    
    query: str
    answer: str
    evidence: list[Evidence]
    confidence: float = Field(ge=0, le=1)


class RAGService:
    """
    RAG-based question answering service.
    
    For now, uses simple extractive summarization.
    Future: LLM-based reasoning.
    """
    
    def __init__(self, top_k: int = 5, min_score: float = 0.3):
        self.top_k = top_k
        self.min_score = min_score
    
    def answer(self, query: str) -> RAGAnswer:
        """
        Answer a question using RAG.
        
        Args:
            query: User question
            
        Returns:
            RAGAnswer with evidence and summary
        """
        # Retrieve relevant documents
        index = get_rag_index()
        results = index.search(query, k=self.top_k)
        
        # Filter by minimum score
        results = [r for r in results if r.get("score", 0) >= self.min_score]
        
        if not results:
            return RAGAnswer(
                query=query,
                answer="No relevant information found. Try indexing more documents or rephrasing your question.",
                evidence=[],
                confidence=0.0,
            )
        
        # Build evidence list
        evidence = []
        for doc in results:
            evidence.append(Evidence(
                id=doc.get("id", ""),
                type=doc.get("type", "unknown"),
                text=doc.get("text", ""),
                score=doc.get("score", 0),
                metadata={
                    k: v for k, v in doc.items()
                    if k not in ["id", "type", "text", "score"]
                },
            ))
        
        # Generate answer (simple extractive for now)
        answer = self._generate_answer(query, results)
        
        # Calculate confidence
        avg_score = sum(r.get("score", 0) for r in results) / len(results)
        confidence = min(avg_score, 1.0)
        
        return RAGAnswer(
            query=query,
            answer=answer,
            evidence=evidence,
            confidence=round(confidence, 3),
        )
    
    def _generate_answer(self, query: str, results: list[dict]) -> str:
        """
        Generate an answer from retrieved results.
        
        Simple extractive approach for now.
        """
        query_lower = query.lower()
        
        # Categorize the question
        if any(w in query_lower for w in ["what model", "which model", "model type"]):
            return self._answer_about_models(results)
        
        elif any(w in query_lower for w in ["training", "train", "trained"]):
            return self._answer_about_training(results)
        
        elif any(w in query_lower for w in ["feature", "indicator", "rsi", "macd"]):
            return self._answer_about_features(results)
        
        elif any(w in query_lower for w in ["performance", "accuracy", "auc", "metric"]):
            return self._answer_about_performance(results)
        
        else:
            # Generic answer
            return self._generic_answer(results)
    
    def _answer_about_models(self, results: list[dict]) -> str:
        """Generate answer about models."""
        model_types = set()
        descriptions = []
        
        for r in results:
            if r.get("type") == "model_doc":
                descriptions.append(r.get("text", ""))
            if "model_type" in r:
                model_types.add(r["model_type"])
        
        if descriptions:
            return " ".join(descriptions[:2])
        elif model_types:
            return f"Available models: {', '.join(model_types)}."
        else:
            return self._generic_answer(results)
    
    def _answer_about_training(self, results: list[dict]) -> str:
        """Generate answer about training."""
        training_runs = [r for r in results if r.get("type") == "training_run"]
        
        if not training_runs:
            return "No training runs found matching your query."
        
        # Summarize recent runs
        summaries = []
        for run in training_runs[:3]:
            tickers = run.get("tickers", [])
            model = run.get("model_type", "unknown")
            success = "✓" if run.get("success") else "✗"
            summaries.append(f"{success} {model} on {', '.join(tickers)}")
        
        return f"Recent training runs: {'; '.join(summaries)}."
    
    def _answer_about_features(self, results: list[dict]) -> str:
        """Generate answer about features."""
        feature_groups = [r for r in results if r.get("type") == "feature_group"]
        
        if feature_groups:
            texts = [r.get("text", "") for r in feature_groups[:2]]
            return " ".join(texts)
        
        return self._generic_answer(results)
    
    def _answer_about_performance(self, results: list[dict]) -> str:
        """Generate answer about performance metrics."""
        training_runs = [r for r in results if r.get("type") == "training_run"]
        
        if not training_runs:
            return "No performance data found."
        
        # Find best performing
        best_run = max(
            training_runs,
            key=lambda r: r.get("metadata", {}).get("metrics", {}).get("val_auc", 0),
            default=None
        )
        
        if best_run:
            return best_run.get("text", "Performance data available in training runs.")
        
        return self._generic_answer(results)
    
    def _generic_answer(self, results: list[dict]) -> str:
        """Generate generic answer from top results."""
        if not results:
            return "No relevant information found."
        
        # Return top result's text
        top_texts = [r.get("text", "") for r in results[:2]]
        return " ".join(top_texts)


# Convenience function
def rag_answer(query: str, top_k: int = 5) -> RAGAnswer:
    """Answer a question using RAG."""
    service = RAGService(top_k=top_k)
    return service.answer(query)

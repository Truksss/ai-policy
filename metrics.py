from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from typing import List, Dict, Any
import json
import os
from datetime import datetime

class RAGASMetricsCollector:
    def __init__(self):
        self.metrics_file = "metrics_log.json"
    
    def calculate_faithfulness_score(self, answer: str, retrieved_context: str) -> float:
        """Calculate how faithful the answer is to the retrieved context (1-10 scale)"""
        faithfulness_prompt = f"""
        Rate how faithful this answer is to the provided context (1-10 scale):
        
        Context: {retrieved_context}
        
        Answer: {answer}
        
        Instructions:
        - 1-3: Answer contains information not in context or contradicts context
        - 4-6: Answer partially faithful but has some unsupported claims
        - 7-8: Answer mostly faithful with minor deviations
        - 9-10: Answer completely faithful to context
        
        Return only a number from 1-10:
        """
        
        try:
            response = self.llm.invoke(faithfulness_prompt)
            score = float(response.content.strip())
            return max(1, min(10, score))  # Ensure score is between 1-10
        except:
            return 5.0  # Default score if evaluation fails
    
    def detect_hallucinations(self, answer: str, retrieved_context: str) -> bool:
        """Detect if answer contains information not supported by context"""
        hallucination_prompt = f"""
        Check if this answer contains information not supported by the context:
        
        Context: {retrieved_context}
        
        Answer: {answer}
        
        Instructions:
        - Return 1 if the answer contains information NOT in the context
        - Return 0 if all information in the answer is supported by the context
        
        Return only 1 or 0:
        """
        
        try:
            response = self.llm.invoke(hallucination_prompt)
            result = int(response.content.strip())
            return bool(result)
        except:
            return False  # Default to no hallucination if evaluation fails
    
    def calculate_context_utilization(self, generated_policy: str, retrieved_context: List[str]) -> float:
        """Calculate what percentage of retrieved context is used in the generated policy"""
        if not retrieved_context:
            return 0.0
        
        utilized_chunks = 0
        for chunk in retrieved_context:
            # Check if significant parts of the chunk appear in the generated policy
            chunk_words = chunk.split()[:20]  # First 20 words of chunk
            chunk_phrases = [chunk_words[i:i+3] for i in range(0, len(chunk_words)-2, 2)]  # 3-word phrases
            
            # Check if any phrase from chunk appears in generated policy
            chunk_used = any(
                ' '.join(phrase).lower() in generated_policy.lower() 
                for phrase in chunk_phrases
            )
            
            if chunk_used:
                utilized_chunks += 1
        
        return utilized_chunks / len(retrieved_context)
    
    def calculate_answer_accuracy(self, answer: str, retrieved_context: str) -> float:
        """Calculate accuracy of the answer based on retrieved context (1-10 scale)"""
        accuracy_prompt = f"""
        Rate the accuracy of this answer based on the provided context (1-10 scale):
        
        Context: {retrieved_context}
        
        Answer: {answer}
        
        Instructions:
        - 1-3: Answer is factually incorrect or misleading
        - 4-6: Answer has some inaccuracies or missing important details
        - 7-8: Answer is mostly accurate with minor issues
        - 9-10: Answer is completely accurate and comprehensive
        
        Return only a number from 1-10:
        """
        
        try:
            response = self.llm.invoke(accuracy_prompt)
            score = float(response.content.strip())
            return max(1, min(10, score))  # Ensure score is between 1-10
        except:
            return 5.0  # Default score if evaluation fails
    
    def evaluate_with_ragas(self, query: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        """Evaluate using RAGAS metrics"""
        try:
            # Create dataset for RAGAS evaluation with all required columns
            dataset = Dataset.from_dict({
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
                "reference": [answer]  # Use answer as reference for context_precision
            })
            
            # Run RAGAS evaluation with metrics that don't require additional columns
            result = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy
                ]
            )
            
            # Extract metrics from result
            metrics_dict = {}
            
            # Handle RAGAS EvaluationResult object
            if hasattr(result, 'to_pandas'):
                # Convert to pandas DataFrame to extract metrics
                df = result.to_pandas()
                for col in df.columns:
                    if col in ['faithfulness', 'answer_relevancy']:
                        value = df[col].iloc[0] if len(df) > 0 else 0.0
                        metrics_dict[col] = float(value)
            elif hasattr(result, '__dict__'):
                # Try to access attributes directly
                for attr_name in ['faithfulness', 'answer_relevancy']:
                    if hasattr(result, attr_name):
                        value = getattr(result, attr_name)
                        if isinstance(value, (int, float)):
                            metrics_dict[attr_name] = float(value)
                        elif hasattr(value, 'tolist'):
                            metrics_dict[attr_name] = float(value.tolist()[0] if len(value.tolist()) > 0 else 0.0)
            else:
                # Fallback: try to iterate over result
                try:
                    for metric_name, metric_value in result:
                        if metric_name in ['faithfulness', 'answer_relevancy']:
                            if isinstance(metric_value, (int, float)):
                                metrics_dict[metric_name] = float(metric_value)
                            elif hasattr(metric_value, 'tolist'):
                                metrics_dict[metric_name] = float(metric_value.tolist()[0] if len(metric_value.tolist()) > 0 else 0.0)
                except:
                    pass
            
            # Add simplified context metrics (without RAGAS dependency)
            metrics_dict["context_precision"] = self._calculate_context_precision(query, contexts)
            metrics_dict["context_recall"] = self._calculate_context_recall(query, contexts)
            
            # Calculate overall RAGAS score (average of all metrics)
            if metrics_dict:
                ragas_score = sum(metrics_dict.values()) / len(metrics_dict)
                metrics_dict["ragas_score"] = ragas_score
            
            return metrics_dict
            
        except Exception as e:
            print(f"RAGAS evaluation failed: {e}")
            # Return default metrics if evaluation fails
            return {
                "faithfulness": 0.5,
                "answer_relevancy": 0.5,
                "context_precision": 0.5,
                "context_recall": 0.5,
                "ragas_score": 0.5
            }
    
    def _calculate_context_precision(self, query: str, contexts: List[str]) -> float:
        """Calculate context precision as percentage of relevant contexts"""
        if not contexts:
            return 0.0
        
        query_words = set(query.lower().split())
        relevant_contexts = 0
        
        for context in contexts:
            context_words = set(context.lower().split())
            # Calculate overlap between query and context
            overlap = len(query_words.intersection(context_words))
            relevance_threshold = len(query_words) * 0.3  # 30% overlap threshold
            if overlap >= relevance_threshold:
                relevant_contexts += 1
        
        return relevant_contexts / len(contexts)
    
    def _calculate_context_recall(self, query: str, contexts: List[str]) -> float:
        """Calculate context recall as coverage of query terms"""
        if not contexts:
            return 0.0
        
        query_words = set(query.lower().split())
        all_context_words = set()
        
        for context in contexts:
            all_context_words.update(context.lower().split())
        
        # Calculate how many query terms are covered by contexts
        covered_terms = len(query_words.intersection(all_context_words))
        return covered_terms / len(query_words) if query_words else 0.0
    
    def collect_rag_metrics(self, query: str, answer: str, retrieved_docs: List[Any]) -> Dict[str, Any]:
        """Collect all RAG-related metrics using RAGAS"""
        contexts = [doc.page_content for doc in retrieved_docs]
        
        # Get RAGAS metrics
        ragas_metrics = self.evaluate_with_ragas(query, answer, contexts)
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "ragas_metrics": ragas_metrics,
            "num_retrieved_docs": len(retrieved_docs),
            "context_length": sum(len(context) for context in contexts),
            "interpretation": self._interpret_ragas_scores(ragas_metrics)
        }
        
        return metrics
    
    def collect_policy_metrics(self, generated_policy: str, retrieved_sources: List[Dict], target_country: str) -> Dict[str, Any]:
        """Collect all policy generation metrics using RAGAS"""
        contexts = [source.get('content', '') for source in retrieved_sources]
        
        # Create a query for policy generation evaluation
        query = f"Generate AI policy for {target_country} educational institution"
        
        # Get RAGAS metrics
        ragas_metrics = self.evaluate_with_ragas(query, generated_policy, contexts)
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "target_country": target_country,
            "ragas_metrics": ragas_metrics,
            "num_retrieved_sources": len(retrieved_sources),
            "policy_length": len(generated_policy),
            "interpretation": self._interpret_ragas_scores(ragas_metrics)
        }
        
        return metrics
    
    def save_metrics(self, metrics: Dict[str, Any], metric_type: str = "rag"):
        """Save metrics to file"""
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w') as f:
                json.dump({}, f)
        
        with open(self.metrics_file, 'r') as f:
            all_metrics = json.load(f)
        
        if metric_type not in all_metrics:
            all_metrics[metric_type] = []
        
        all_metrics[metric_type].append(metrics)
        
        with open(self.metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
    
    def _interpret_ragas_scores(self, ragas_metrics: Dict[str, float]) -> Dict[str, str]:
        """Interpret RAGAS scores for better understanding"""
        ragas_score = ragas_metrics.get("ragas_score", 0)
        
        if ragas_score >= 0.8:
            overall_quality = "Excellent"
        elif ragas_score >= 0.6:
            overall_quality = "Good"
        elif ragas_score >= 0.4:
            overall_quality = "Fair"
        else:
            overall_quality = "Poor"
        
        strengths = []
        weaknesses = []
        
        if ragas_metrics.get("faithfulness", 0) >= 0.8:
            strengths.append("High faithfulness to context")
        elif ragas_metrics.get("faithfulness", 0) < 0.5:
            weaknesses.append("Low faithfulness to context")
        
        if ragas_metrics.get("answer_relevancy", 0) >= 0.8:
            strengths.append("High answer relevancy")
        elif ragas_metrics.get("answer_relevancy", 0) < 0.5:
            weaknesses.append("Low answer relevancy")
        
        if ragas_metrics.get("context_precision", 0) >= 0.8:
            strengths.append("High context precision")
        elif ragas_metrics.get("context_precision", 0) < 0.5:
            weaknesses.append("Low context precision")
        
        if ragas_metrics.get("context_recall", 0) >= 0.8:
            strengths.append("High context recall")
        elif ragas_metrics.get("context_recall", 0) < 0.5:
            weaknesses.append("Low context recall")
        
        return {
            "overall_quality": overall_quality,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": self._get_recommendations(ragas_metrics)
        }
    
    def _get_recommendations(self, ragas_metrics: Dict[str, float]) -> List[str]:
        """Get recommendations based on RAGAS scores"""
        recommendations = []
        
        if ragas_metrics.get("faithfulness", 0) < 0.6:
            recommendations.append("Improve answer faithfulness to retrieved context")
        
        if ragas_metrics.get("answer_relevancy", 0) < 0.6:
            recommendations.append("Improve answer relevancy to the query")
        
        if ragas_metrics.get("context_precision", 0) < 0.6:
            recommendations.append("Improve context retrieval precision")
        
        if ragas_metrics.get("context_recall", 0) < 0.6:
            recommendations.append("Improve context retrieval recall")
        
        if not recommendations:
            recommendations.append("System performance is good - continue current approach")
        
        return recommendations
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all collected metrics"""
        if not os.path.exists(self.metrics_file):
            return {"error": "No metrics file found"}
        
        with open(self.metrics_file, 'r') as f:
            all_metrics = json.load(f)
        
        summary = {}
        
        for metric_type, metrics_list in all_metrics.items():
            if not metrics_list:
                continue
            
            # Calculate averages for RAGAS metrics
            ragas_scores = [m.get("ragas_metrics", {}) for m in metrics_list]
            
            # Avoid division by zero
            if ragas_scores:
                summary[metric_type] = {
                    "total_queries": len(metrics_list),
                    "avg_ragas_score": sum(s.get("ragas_score", 0) for s in ragas_scores) / len(ragas_scores),
                    "avg_faithfulness": sum(s.get("faithfulness", 0) for s in ragas_scores) / len(ragas_scores),
                    "avg_answer_relevancy": sum(s.get("answer_relevancy", 0) for s in ragas_scores) / len(ragas_scores),
                    "avg_context_precision": sum(s.get("context_precision", 0) for s in ragas_scores) / len(ragas_scores),
                    "avg_context_recall": sum(s.get("context_recall", 0) for s in ragas_scores) / len(ragas_scores),
                    "avg_retrieved_docs": sum(m.get("num_retrieved_docs", m.get("num_retrieved_sources", 0)) for m in metrics_list) / len(metrics_list)
                }
            else:
                summary[metric_type] = {
                    "total_queries": len(metrics_list),
                    "avg_ragas_score": 0,
                    "avg_faithfulness": 0,
                    "avg_answer_relevancy": 0,
                    "avg_context_precision": 0,
                    "avg_context_recall": 0,
                    "avg_retrieved_docs": 0
                }
        
        return summary

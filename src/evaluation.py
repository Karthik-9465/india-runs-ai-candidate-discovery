# src/evaluation.py
import math

class RankingEvaluator:
    """Offline Evaluation Framework that calculates NDCG, MRR, MAP, Precision@K, and Recall@K."""
    
    def __init__(self):
        pass

    @staticmethod
    def calculate_dcg(relevances, k):
        """Calculates Discounted Cumulative Gain (DCG) at K."""
        dcg = 0.0
        for i in range(min(len(relevances), k)):
            rel = relevances[i]
            # Standard formula: (2^rel - 1) / log2(i + 2)
            dcg += (2**rel - 1) / math.log2(i + 2)
        return dcg

    def calculate_ndcg(self, ranked_ids, ground_truth, k):
        """Calculates Normalized Discounted Cumulative Gain (NDCG) at K."""
        # Get relevance grades for our ranked candidates
        relevances = [ground_truth.get(cid, 0.0) for cid in ranked_ids[:k]]
        actual_dcg = self.calculate_dcg(relevances, k)
        
        # Calculate Ideal DCG (IDCG) by sorting all ground truth relevance grades descending
        ideal_relevances = sorted([val for val in ground_truth.values()], reverse=True)[:k]
        ideal_dcg = self.calculate_dcg(ideal_relevances, k)
        
        if ideal_dcg == 0.0:
            return 0.0
        return actual_dcg / ideal_dcg

    @staticmethod
    def calculate_mrr(ranked_ids, ground_truth, threshold=3.0):
        """Calculates Mean Reciprocal Rank (MRR) based on the first relevant item (grade >= threshold)."""
        for rank_idx, cid in enumerate(ranked_ids):
            rel = ground_truth.get(cid, 0.0)
            if rel >= threshold:
                return 1.0 / (rank_idx + 1)
        return 0.0

    @staticmethod
    def calculate_map(ranked_ids, ground_truth, threshold=3.0):
        """Calculates Mean Average Precision (MAP) based on relevant items (grade >= threshold)."""
        relevant_count = 0
        precision_sum = 0.0
        
        # Count total relevant items in ground truth
        total_relevant_in_gt = sum(1 for rel in ground_truth.values() if rel >= threshold)
        if total_relevant_in_gt == 0:
            return 0.0
            
        for rank_idx, cid in enumerate(ranked_ids):
            rel = ground_truth.get(cid, 0.0)
            if rel >= threshold:
                relevant_count += 1
                # Precision at this rank
                precision_at_i = relevant_count / (rank_idx + 1)
                precision_sum += precision_at_i
                
        return precision_sum / total_relevant_in_gt

    @staticmethod
    def calculate_precision_at_k(ranked_ids, ground_truth, k, threshold=3.0):
        """Calculates Precision@K (fraction of top K candidates that are relevant, grade >= threshold)."""
        if k == 0:
            return 0.0
        relevant_count = sum(1 for cid in ranked_ids[:k] if ground_truth.get(cid, 0.0) >= threshold)
        return relevant_count / k

    @staticmethod
    def calculate_recall_at_k(ranked_ids, ground_truth, k, threshold=3.0):
        """Calculates Recall@K (fraction of all relevant candidates that are captured in top K)."""
        total_relevant_in_gt = sum(1 for rel in ground_truth.values() if rel >= threshold)
        if total_relevant_in_gt == 0:
            return 1.0  # Perfect recall if there are no relevant items
            
        relevant_captured = sum(1 for cid in ranked_ids[:k] if ground_truth.get(cid, 0.0) >= threshold)
        return relevant_captured / total_relevant_in_gt
        
    def evaluate_all(self, ranked_ids, ground_truth, threshold=3.0):
        """Computes a dictionary of all evaluation metrics."""
        return {
            "NDCG@10": round(self.calculate_ndcg(ranked_ids, ground_truth, 10), 4),
            "NDCG@50": round(self.calculate_ndcg(ranked_ids, ground_truth, 50), 4),
            "MAP": round(self.calculate_map(ranked_ids, ground_truth, threshold), 4),
            "Precision@5": round(self.calculate_precision_at_k(ranked_ids, ground_truth, 5, threshold), 4),
            "Precision@10": round(self.calculate_precision_at_k(ranked_ids, ground_truth, 10, threshold), 4),
            "Recall@50": round(self.calculate_recall_at_k(ranked_ids, ground_truth, 50, threshold), 4),
            "Recall@100": round(self.calculate_recall_at_k(ranked_ids, ground_truth, 100, threshold), 4),
            "MRR": round(self.mrr_eval(ranked_ids, ground_truth, threshold), 4)
        }
        
    def mrr_eval(self, ranked_ids, ground_truth, threshold=3.0):
        return self.calculate_mrr(ranked_ids, ground_truth, threshold)

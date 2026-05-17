"""
Evaluation Metrics Module

Implements NLG metrics (BLEU, METEOR, ROUGE-L) and Clinical Efficacy metrics
as described in the Reg2RG paper.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import re


class NLGMetrics:
    """Natural Language Generation metrics for report evaluation."""
    
    def __init__(self):
        """Initialize NLG metrics."""
        try:
            import nltk
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from nltk.translate.meteor_score import meteor_score
            
            # Do not download data here. Training/evaluation often runs in
            # offline environments; METEOR is skipped when WordNet is absent.
            self.meteor_available = True
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                self.meteor_available = False
            
            self.nltk = nltk
            self.sentence_bleu = sentence_bleu
            self.meteor_score = meteor_score
            self.smoothing = SmoothingFunction()
            
        except ImportError:
            raise ImportError("Please install nltk: pip install nltk")
    
    def compute_bleu(
        self,
        references: List[str],
        hypotheses: List[str],
        max_n: int = 4
    ) -> Dict[str, float]:
        """
        Compute BLEU-1 to BLEU-4 scores.
        
        Args:
            references: List of reference texts
            hypotheses: List of generated texts
            max_n: Maximum n-gram (default 4)
        
        Returns:
            Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
        """
        scores = {f"BLEU-{i}": [] for i in range(1, max_n + 1)}
        
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = [ref.lower().split()]
            hyp_tokens = hyp.lower().split()
            
            for n in range(1, max_n + 1):
                weights = tuple([1.0 / n] * n + [0.0] * (max_n - n))
                score = self.sentence_bleu(
                    ref_tokens,
                    hyp_tokens,
                    weights=weights,
                    smoothing_function=self.smoothing.method1
                )
                scores[f"BLEU-{n}"].append(score)
        
        # Average scores
        return {k: np.mean(v) for k, v in scores.items()}
    
    def compute_meteor(
        self,
        references: List[str],
        hypotheses: List[str]
    ) -> Optional[float]:
        """
        Compute METEOR score.
        
        Args:
            references: List of reference texts
            hypotheses: List of generated texts
        
        Returns:
            Average METEOR score
        """
        if not self.meteor_available:
            return None

        scores = []
        
        for ref, hyp in zip(references, hypotheses):
            score = self.meteor_score([ref.lower().split()], hyp.lower().split())
            scores.append(score)
        
        return np.mean(scores)
    
    def compute_rouge_l(
        self,
        references: List[str],
        hypotheses: List[str]
    ) -> float:
        """
        Compute ROUGE-L score.
        
        Args:
            references: List of reference texts
            hypotheses: List of generated texts
        
        Returns:
            Average ROUGE-L F1 score
        """
        def lcs_length(x, y):
            """Compute longest common subsequence length."""
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            
            return dp[m][n]
        
        scores = []
        
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = ref.lower().split()
            hyp_tokens = hyp.lower().split()
            
            lcs_len = lcs_length(ref_tokens, hyp_tokens)
            
            if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
                scores.append(0.0)
                continue
            
            precision = lcs_len / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
            recall = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            
            scores.append(f1)
        
        return np.mean(scores)
    
    def compute_all(
        self,
        references: List[str],
        hypotheses: List[str]
    ) -> Dict[str, float]:
        """
        Compute all NLG metrics.
        
        Args:
            references: List of reference texts
            hypotheses: List of generated texts
        
        Returns:
            Dictionary with all metric scores
        """
        results = {}
        
        # BLEU scores
        bleu_scores = self.compute_bleu(references, hypotheses)
        results.update(bleu_scores)
        
        # METEOR
        results["METEOR"] = self.compute_meteor(references, hypotheses)
        
        # ROUGE-L
        results["ROUGE-L"] = self.compute_rouge_l(references, hypotheses)
        
        return results


class ClinicalEfficacyMetrics:
    """
    Clinical Efficacy (CE) metrics for LNDb dataset.
    
    Extracts clinical labels from reports and computes precision/recall/F1.
    """
    
    def __init__(self):
        """Initialize clinical efficacy metrics."""
        # Define label extraction patterns
        self.label_patterns = {
            'nodule_present': r'\b(nodule|mass|lesion)\b',
            'size_small': r'\b(small|<\s*6\s*mm|less than 6)\b',
            'size_medium': r'\b(6\s*-\s*10\s*mm|medium)\b',
            'size_large': r'\b(>\s*10\s*mm|large|greater than 10)\b',
            'solid': r'\bsolid\b',
            'subsolid': r'\bsubsolid\b',
            'ggo': r'\b(ground.glass|GGO)\b',
            'calcification': r'\bcalcif',
            'spiculation': r'\bspiculat'
        }
    
    def extract_labels(self, text: str) -> Dict[str, bool]:
        """
        Extract binary labels from report text.
        
        Args:
            text: Report text
        
        Returns:
            Dictionary of label -> bool
        """
        text_lower = text.lower()
        labels = {}
        
        for label_name, pattern in self.label_patterns.items():
            labels[label_name] = bool(re.search(pattern, text_lower))
        
        return labels
    
    def compute_metrics(
        self,
        references: List[str],
        hypotheses: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute precision, recall, F1 for each label.
        
        Args:
            references: List of reference reports
            hypotheses: List of generated reports
        
        Returns:
            Dictionary mapping label -> {precision, recall, f1}
        """
        # Extract labels
        ref_labels_list = [self.extract_labels(ref) for ref in references]
        hyp_labels_list = [self.extract_labels(hyp) for hyp in hypotheses]
        
        # Compute metrics per label
        results = {}
        
        for label_name in self.label_patterns.keys():
            tp = sum(
                ref_labels[label_name] and hyp_labels[label_name]
                for ref_labels, hyp_labels in zip(ref_labels_list, hyp_labels_list)
            )
            fp = sum(
                not ref_labels[label_name] and hyp_labels[label_name]
                for ref_labels, hyp_labels in zip(ref_labels_list, hyp_labels_list)
            )
            fn = sum(
                ref_labels[label_name] and not hyp_labels[label_name]
                for ref_labels, hyp_labels in zip(ref_labels_list, hyp_labels_list)
            )
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            results[label_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        # Compute macro-average
        all_precisions = [v['precision'] for v in results.values()]
        all_recalls = [v['recall'] for v in results.values()]
        all_f1s = [v['f1'] for v in results.values()]
        
        results['macro_avg'] = {
            'precision': np.mean(all_precisions),
            'recall': np.mean(all_recalls),
            'f1': np.mean(all_f1s)
        }
        
        return results

"""
Document AI detection service:

Text from the document is first extracted, then five signals are computed on the raw text.
"""

import asyncio
import io
import logging
import math
import re
import string
from collections import Counter
from typing import Optional

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import PyPDF2
import docx

from app.core.config import get_settings
from app.features.document.schemas import(
    ConfidenceLevel,
    DocumentSignals,
    DocumentDetectionResult,
)

logger = logging.getLogger(__name__)
settings = get_settings()

# Maximum number of tokens the RoBERTa model can handle.
MAX_TOKENS = 512
# Maximum number of characters we will analyze (long docs get truncated).
MAX_CHARS = 10_000

class DocumentDetectionService:
    _WEIGHTS = {
        "perplexity": 0.15,
        "burstiness": 0.15,
        "stylometric": 0.15,
        "repetition": 0.15,
        "classifier": 0.40, # Most reliable.
    }

    def __init__(self) -> None:
        self._ort_session = None
        self._tokenizer = None
        self._models_loaded = False
    
    def load_models(self) -> None:
        """Load ONNX model and tokenizer. Called once at startup."""
        logger.info("Loading document detection models...")

        self._ort_session = ort.InferenceSession(settings.text_classifier_model)
        logger.info("Text classifier ONNX model loaded from %s", settings.text_classifier_model)

        self._tokenizer = AutoTokenizer.from_pretrained(settings.text_classifier_tokenizer_dir)
        logger.info("Tokenizer loaded from %s", settings.text_classifier_tokenizer_dir)

        self._models_loaded = True
        logger.info("Document detection models loaded successfully")

    # ----------------------------------------------------------------------------------------------
    # Entry point
    # ----------------------------------------------------------------------------------------------

    async def analyze(self, contents: bytes, content_type: str) -> DocumentDetectionResult:
        # Step 1 - extract text from whatever format was uploaded.
        text = self._extract_text(contents, content_type)

        word_count = len(text.split())
        char_count = len(text)

        # Truncate very long docs - signals become unreliable on huge texts.
        truncated = len(text) > MAX_CHARS
        if truncated:
            text = text[:MAX_CHARS]
            logger.info("Document truncated to %d chars for analysis", MAX_CHARS)
        
        if word_count < 20:
            logger.warning("Document too short for reliable detection - word_count=%d", word_count)
        
        # Step 2 - run all signals concurrently.
        loop = asyncio.get_event_loop()
        perplexity_score, burstiness_score, stylometric_score, repetition_score, classifier_score = await asyncio.gather(
            loop.run_in_executor(None, self._perplexity_score, text),
            loop.run_in_executor(None, self._burstiness_score, text),
            loop.run_in_executor(None, self._stylometric_score, text),
            loop.run_in_executor(None, self._repetition_score, text),
            loop.run_in_executor(None, self._classifier_score, text),
        )

        signals = DocumentSignals(
            perplexity_score=round(perplexity_score, 4),
            burstiness_score=round(burstiness_score, 4),
            stylometric_score=round(stylometric_score, 4),
            repetition_score=round(repetition_score, 4),
            classifier_score=round(classifier_score, 4),
        )

        ai_probability = self._combine(signals)
        confidence = self._confidence(signals)
        verdict = self._verdict(ai_probability)

        logger.info(
            "document_analysis_complete perplexity=%.3f burstiness=%.3f "
            "stylometric=%.3f repetition=%.3f classifier=%.3f final=%.3f",
            perplexity_score, burstiness_score, stylometric_score,
            repetition_score, classifier_score, ai_probability,
        )

        return DocumentDetectionResult(
            ai_probability=round(ai_probability, 4),
            confidence=confidence,
            signals=signals,
            verdict=verdict,
            word_count=word_count,
            char_count=char_count,
            truncated=truncated,
        )

    # ----------------------------------------------------------------------------------------------
    # Text extraction - format specific
    # ----------------------------------------------------------------------------------------------

    def _extract_text(self, contents: bytes, content_type: str) -> str:
        if content_type == "text/plain":
            return self._extract_txt(contents)
        elif content_type == "application/pdf":
            return self._extract_pdf(contents)
        elif "wordprocessingml" in content_type: # Covers both .doc and .docx
            return self._extract_docx(contents)
        else:
            # Fallback - try plain text.
            return contents.decode("utf-8", errors="ignore")
    
    def _extract_txt(self, contents: bytes) -> str:
        return contents.decode("utf-8", errors="ignore")
    
    def _extract_pdf(self, contents: bytes) -> str:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(contents))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text.strip())
            return "\n\n".join(pages)
        except Exception as e:
            logger.error("PDF extraction failed: %s", e)
            return ""
    
    def _extract_docx(self, contents: bytes) -> str:
        try:
            doc = docx.Document(io.BytesIO(contents))
            paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paras)
        except Exception as e:
            logger.error("DOCX extraction failed: %s", e)
            return ""
    
    # ----------------------------------------------------------------------------------------------
    # Signal 1 - Perplexity
    # ----------------------------------------------------------------------------------------------

    def _perplexity_score(self, text: str) -> float:
        """
        Approximates perplexity using bigram probability. Bigram probability is a simple measure of how predictable the text is.
        AI text is unnaturally predictable - low perplexity. Returns 0.0-1.0 = string AI signal.
        """
        words = self._tokenize_words(text)
        if len(words) < 10:
            return 0.5 # Not enough data for reliable signal - return neutral score.
        
        # Build unigram and bigram counts.
        unigrams = Counter(words)
        bigrams = Counter(zip(words[:-1], words[1:]))
        total = len(words)

        # Compute average bigram log probability.
        log_probs = []
        for w1, w2 in zip(words[:-1], words[1:]):
            unigram_prob = unigrams[w1] / total
            bigram_prob = (bigrams[(w1, w2)] + 1e-6) / (unigrams[w1] + 1e-6)
            log_probs.append(math.log(bigram_prob + 1e-10))
        
        avg_log_prob = float(np.mean(log_probs))

        # More negative avg_log_prob = more surprising = more human-like.
        # Less negative (closer to 0) = more predictable = more AI-like.
        # Sigmoid maps this to 0-1 where higher = more AI.
        score  = self._sigmoid(-avg_log_prob * 2 - 3) # Tuned on sample data - adjust multiplier and shift as needed.
        return float(np.clip(score, 0.0, 1.0))
    
    # ----------------------------------------------------------------------------------------------
    # Signal 2 - Burstiness
    # ----------------------------------------------------------------------------------------------

    def _burstiness_score(self, text: str) -> float:
        """
        Measures sentence length variance. Human writing is bursty - high varaiance in sentence lengths.
        AI writing is unnaturally uniform. Returns 0.0-1.0 where 1.0 = strong AI signal.
        """
        sentences = self._split_sentences(text)
        if len(sentences) < 3:
            return 0.5 # Not enough data for reliable signal - return neutral score.
        
        lengths = np.array([len(s.split()) for s in sentences], dtype=float)
        mean = float(np.mean(lengths))
        std = float(np.std(lengths))

        if mean == 0:
            return 0.5 # Avoid division by zero - treat as neutral.
        
        # Coefficient of variation - lower = more uniform = more AI-like.
        cv = std / mean

        # Real himan text: cv typically > 0.5
        # AI text: cv typically < 0.3
        score = self._sigmoid(-cv * 6 + 2)
        return float(np.clip(score, 0.0, 1.0))
    
    # ----------------------------------------------------------------------------------------------
    # Signal 3 - Stylometric analysis
    # ----------------------------------------------------------------------------------------------
    def _stylometric_score(self, text: str) -> float:
        """
        Analyse writing style fingerprints:
        - Vocab richness (type-token ratio)
        - Function word overuse
        - Punctuation patterns
        - Average word length
        Returnts 0.0-1.0 where 1.0 = strong AI signal.
        """

        words = self._tokenize_words(text) # Simple word tokenizer - can be improved.
        if len(words) < 20:
            return 0.5 # Not enough data for reliable signal - return neutral score.
        
        scores = []

        # --- Vocab richness ---
        # AI tends to reuse the same vocab more than humans do.
        unique_ratio = len(set(words)) / len(words)

        # Human text: typically 0.4-0.7 unique ratio.
        # AI text: typically < 0.35 (repetitive) or very high (unnaturalkly varied).
        richness_anomaly = abs(unique_ratio - 0.52) / 0.52 # Tuned on sample data - adjust as needed.
        scores.append(float(np.clip(richness_anomaly * 2, 0.0, 1.0))) # Scale to 0-1 where higher = more AI-like.

        # --- Function word overuse ---
        # AI overuses certain connectives like "furthermore", "moreover", "additionally", etc.
        AI_CONNECTIVES = {
            "furthermore", "moreover", "additionally", "consequently",
            "nevertheless", "nonetheless", "therefore", "thus",
            "subsequently", "accordingly", "henceforth"
        }
        word_set = set(w.lower() for w in words)
        connective_hits = len(word_set & AI_CONNECTIVES)
        connective_score = min(connective_hits / 3, 1.0) # 3 or more hits = strong AI signal.
        scores.append(connective_score)

        # --- Average word length ---
        # AI tends towards slightly longer, more formal words.
        avg_word_len = float(np.mean([len(w) for w in words]))

        # Human casual writing: ~4.5 chars, AI formal: ~5.5+ chars.
        word_len_score = self._sigmoid(avg_word_len - 5.5)
        scores.append(word_len_score)

        # --- Punctuation consistency ---
        # Count punctuation marks per sentence.
        sentences = self._split_sentences(text)
        punct_per_sent = [sum(1 for c in s if c in string.punctuation) for s in sentences]
        if len(punct_per_sent) > 1:
            punct_cv = np.std(punct_per_sent) / (np.mean(punct_per_sent) + 1e-6)
            # AI is more consistent with punctuation - lower variance.
            punct_score = self._sigmoid(-float(punct_cv) * 3 + 1)
            scores.append(punct_score)
        
        return float(np.clip(np.mean(scores), 0.0, 1.0))
    
    # ----------------------------------------------------------------------------------------------
    # Signal 4 - N-gram repetition
    # ----------------------------------------------------------------------------------------------
    
    def _repetition_score(self, text: str) -> float:
        """
        Measures how often n-grams repeat across the doc. AI models draw from the same distributions repeatedly,
        producting more repeated phrases than in human writing. Returns 0.0-1.0 where 1.0 = strong AI signal.
        """
        words = self._tokenize_words(text)
        if len(words) < 20:
            return 0.5 # Not enough data for reliable signal - return neutral score.
        
        scores = []
        for n in [2, 3, 4]:
            ngrams = list(zip(*[words[i:] for i in range(n)]))
            if not ngrams:
                continue
            counts = Counter(ngrams)
            total = len(ngrams)
            unique = len(counts)

            # Repetation ratio - higher means more repeated phrases.
            repetition = 1.0 - (unique / total)
            scores.append(repetition)

        if not scores:
            return 0.5 # No n-grams - treat as neutral.
        
        avg_repetition = float(np.mean(scores))
        # Scale - AI text typically has 20-40% repetition at bigram level.
        score = self._sigmoid(avg_repetition * 8 -2)
        return float(np.clip(score, 0.0, 1.0))
    
    # ----------------------------------------------------------------------------------------------
    # Signal 5 - RoBERTa ONNX classifier.
    # ----------------------------------------------------------------------------------------------

    def _classifier_score(self, text: str) -> float:
        """
        Runs text through the fine-tuned RoBERTa ONNX mode. We have installed the ONNX Runtime and
        loaded the model at startup, so this is just a matter of tokenizing the text and running inference.
        Label 0 = Fake (AI), label 1 = Real (Human). Returns 0.0-1.0 where 1.0 = strong AI signal.
        """
        if not self._models_loaded or self._ort_session is None:
            logger.warning("Text classifier not loaded - returning neutral score.")
            return 0.5
        
        # Tokenize - truncate to MAX_TOKENS which is the max the model can handle. Truncation is not ideal but necessary for very long docs.
        encoded = self._tokenizer(
            text,
            max_length=MAX_TOKENS,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )

        outputs = self._ort_session.run(
            ["logits"],
            {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }
        )

        logits = outputs[0][0] # Get logits for the single input.
        exp_logits = np.exp(logits - np.max(logits)) # For numerical stability.
        probs = exp_logits / exp_logits.sum()

        # Index 0 = "Fake" (AI generated).
        ai_score = float(probs[0])
        return float(np.clip(ai_score, 0.0, 1.0))
    
    # ----------------------------------------------------------------------------------------------
    # Combination + helpers
    # ----------------------------------------------------------------------------------------------

    def _combine(self, signals: DocumentSignals) -> float:
        w = self._WEIGHTS
        return float(
            signals.perplexity_score * w["perplexity"] +
            signals.burstiness_score * w["burstiness"] +
            signals.stylometric_score * w["stylometric"] +
            signals.repetition_score * w["repetition"] +
            signals.classifier_score * w["classifier"]
        )
    
    def _confidence(self, signals: DocumentSignals) -> ConfidenceLevel:
        scores = [
            signals.perplexity_score,
            signals.burstiness_score,
            signals.stylometric_score,
            signals.repetition_score,
            signals.classifier_score,
        ]

        variance = float(np.var(scores))
        if variance < 0.02:
            return ConfidenceLevel.HIGH
        elif variance < 0.08:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _verdict(self, probability: float) -> str:
        if probability >= settings.ai_verdict_threshold:
            return "AI-generated"
        elif probability >= settings.possible_ai_verdict_threshold:
            return "Possibly AI-generated"
        else:
            return "Human-generated"
    
    def _tokenize_words(self, text: str) -> list[str]:
        """
        Lowercase, strip punctuation, split into words.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return [w for w in text.split() if w]
    
    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences on . ! ?
        """
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))
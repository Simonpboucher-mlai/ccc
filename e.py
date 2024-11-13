import os
import re
import time
import json
import yaml
import openai
import numpy as np
import tiktoken
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from threading import Lock
from tqdm import tqdm

# Configuration des classes de données
@dataclass
class ChunkMetadata:
    filename: str
    chunk_id: int
    token_count: int
    start_position: int
    end_position: int

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]

@dataclass
class ProcessingStats:
    start_time: float = field(default_factory=time.time)
    processed_files: int = 0
    total_chunks: int = 0
    failed_chunks: int = 0
    embedding_errors: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    processing_times: List[float] = field(default_factory=list)

    def add_file_stats(self, duration: float, chunks: int, failures: int) -> None:
        self.processed_files += 1
        self.total_chunks += chunks
        self.failed_chunks += failures
        self.processing_times.append(duration)

    def add_error(self, error_type: str) -> None:
        self.embedding_errors[error_type] += 1

    def get_summary(self) -> Dict[str, Any]:
        elapsed_time = time.time() - self.start_time
        return {
            'total_files': self.processed_files,
            'total_chunks': self.total_chunks,
            'failed_chunks': self.failed_chunks,
            'success_rate': (self.total_chunks - self.failed_chunks) / self.total_chunks if self.total_chunks else 0,
            'avg_processing_time': sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
            'total_duration': elapsed_time,
            'errors_by_type': dict(self.embedding_errors)
        }

class ValidationManager:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config

    def validate_embedding_format(self, embedding: np.ndarray) -> ValidationResult:
        errors = []
        warnings = []

        if not isinstance(embedding, np.ndarray):
            errors.append("L'embedding n'est pas un numpy array")
            return ValidationResult(False, errors, warnings)

        if embedding.ndim != 1:
            errors.append(f"Dimension incorrecte de l'embedding: {embedding.ndim}")

        if np.isnan(embedding).any():
            errors.append("L'embedding contient des valeurs NaN")

        if np.isinf(embedding).any():
            errors.append("L'embedding contient des valeurs infinies")

        if embedding.shape[0] != self.config['validation']['embedding_size']:
            warnings.append(f"Taille d'embedding inattendue: {embedding.shape[0]}")

        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_chunk_format(self, chunk: Dict[str, Any]) -> ValidationResult:
        errors = []
        warnings = []

        required_fields = ['text', 'metadata']
        required_metadata = ['filename', 'chunk_id', 'token_count']

        for field in required_fields:
            if field not in chunk:
                errors.append(f"Champ requis manquant: {field}")

        if 'metadata' in chunk:
            for field in required_metadata:
                if field not in chunk['metadata']:
                    errors.append(f"Métadonnée requise manquante: {field}")

        if 'text' in chunk:
            text_length = len(chunk['text'].strip())
            if text_length < self.config['validation']['min_chunk_length']:
                warnings.append("Le texte du chunk est trop court")
            elif text_length > self.config['validation']['max_chunk_length']:
                warnings.append("Le texte du chunk est trop long")

        return ValidationResult(len(errors) == 0, errors, warnings)

class CacheManager:
    def __init__(self, cache_dir: str, logger):
        self.cache_dir = cache_dir
        self.logger = logger
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        cache_key = self._get_cache_key(text)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.npy")

        if os.path.exists(cache_file):
            try:
                embedding = np.load(cache_file)
                self.logger.debug(f"Cache hit pour {cache_key}")
                return embedding
            except Exception as e:
                self.logger.warning(f"Erreur lors de la lecture du cache: {e}")
                return None
        return None

    def cache_embedding(self, text: str, embedding: np.ndarray) -> None:
        cache_key = self._get_cache_key(text)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.npy")

        try:
            np.save(cache_file, embedding)
            self.logger.debug(f"Embedding mis en cache: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Erreur lors de la mise en cache: {e}")

class RateLimiter:
    def __init__(self, requests_per_minute: int, logger):
        self.requests_per_minute = requests_per_minute
        self.logger = logger
        self.request_times = deque()
        self.lock = Lock()

    def wait_if_needed(self) -> None:
        with self.lock:
            now = time.time()

            while self.request_times and now - self.request_times[0] > 60:
                self.request_times.popleft()

            if len(self.request_times) >= self.requests_per_minute:
                wait_time = 60 - (now - self.request_times[0])
                if wait_time > 0:
                    self.logger.info(f"Rate limit atteint, attente de {wait_time:.2f} secondes")
                    time.sleep(wait_time)

            self.request_times.append(now)

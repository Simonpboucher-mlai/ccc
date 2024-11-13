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

class TextProcessor:
    def __init__(self, max_tokens: int, header_tokens: int, logger):
        self.max_tokens = max_tokens
        self.header_tokens = header_tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.logger = logger

    def preprocess_text(self, text: str) -> str:
        # Suppression des caractères spéciaux
        text = re.sub(r'[^\w\s\.\,\!\?\-\']', ' ', text)
        
        # Normalisation des espaces et sauts de ligne
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Normalisation de la ponctuation
        text = re.sub(r'\s+([\.,:;!?])', r'\1', text)
        
        return text.strip()

    def get_token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def split_into_chunks_with_header(self, text: str, filename: str) -> List[Tuple[str, ChunkMetadata]]:
        tokens = self.tokenizer.encode(text)
        header = tokens[:self.header_tokens]
        header_text = self.tokenizer.decode(header)
        
        chunks_with_metadata = []
        position = 0
        
        for i, start in enumerate(range(0, len(tokens), self.max_tokens - self.header_tokens)):
            chunk_tokens = header + tokens[start:start + (self.max_tokens - self.header_tokens)]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunk_start = position
            chunk_end = position + len(chunk_text)
            position = chunk_end
            
            metadata = ChunkMetadata(
                filename=filename,
                chunk_id=i,
                token_count=len(chunk_tokens),
                start_position=chunk_start,
                end_position=chunk_end
            )
            
            chunks_with_metadata.append((chunk_text, metadata))
            
        return chunks_with_metadata

    def analyze_text_complexity(self, text: str) -> Dict[str, float]:
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        return {
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'token_density': self.get_token_count(text) / len(words) if words else 0
        }

class EmbeddingGenerator:
    def __init__(self, config: Dict[str, Any], logger, cache_manager, rate_limiter, validation_manager):
        self.model = config['openai']['model']
        self.max_retries = config['openai']['max_retries']
        self.sleep_time = config['openai']['sleep_time']
        self.logger = logger
        self.cache_manager = cache_manager
        self.rate_limiter = rate_limiter
        self.validation_manager = validation_manager

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        # Vérifie d'abord le cache
        cached_embedding = self.cache_manager.get_cached_embedding(text)
        if cached_embedding is not None:
            validation_result = self.validation_manager.validate_embedding_format(cached_embedding)
            if validation_result.is_valid:
                return cached_embedding
            self.logger.warning("Embedding en cache invalide")
            
        # Attend si nécessaire pour respecter les limites
        self.rate_limiter.wait_if_needed()
        
        # Génère un nouvel embedding
        for attempt in range(self.max_retries):
            try:
                response = openai.Embedding.create(
                    input=text,
                    model=self.model
                )
                embedding = np.array(response['data'][0]['embedding'])
                
                # Valide l'embedding
                validation_result = self.validation_manager.validate_embedding_format(embedding)
                if not validation_result.is_valid:
                    raise ValueError(f"Embedding invalide: {validation_result.errors}")
                    
                # Met en cache le nouvel embedding
                self.cache_manager.cache_embedding(text, embedding)
                return embedding
                
            except Exception as e:
                self.logger.error(f"Erreur lors de l'obtention de l'embedding: {e}")
                if attempt < self.max_retries - 1:
                    wait = self.sleep_time * (2 ** attempt)
                    self.logger.info(f"Retry {attempt + 1}/{self.max_retries} après {wait} secondes...")
                    time.sleep(wait)
                else:
                    self.logger.error(f"Échec après {self.max_retries} tentatives.")
                    
        return None

class EmbeddingPipeline:
    def __init__(self, config_path: str = "config/settings.yaml"):
        # Chargement de la configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Configuration du logging
        logging.basicConfig(
            filename=self.config['logging']['filename'],
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            datefmt=self.config['logging']['date_format']
        )
        self.logger = logging.getLogger()

        # Vérification de la clé API OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("La clé API OpenAI n'est pas définie dans les variables d'environnement")
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # Initialisation des composants
        self.cache_manager = CacheManager(self.config['output']['cache_dir'], self.logger)
        self.rate_limiter = RateLimiter(self.config['openai']['requests_per_minute'], self.logger)
        self.validation_manager = ValidationManager(self.logger, self.config)
        self.text_processor = TextProcessor(
            self.config['openai']['max_tokens'],
            self.config['openai']['header_tokens'],
            self.logger
        )
        self.embedding_generator = EmbeddingGenerator(
            self.config,
            self.logger,
            self.cache_manager,
            self.rate_limiter,
            self.validation_manager
        )
        self.stats = ProcessingStats()

    def process_file(self, file_path: str) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
        start_time = time.time()
        filename = os.path.basename(file_path)
        
        try:
            # Lecture et prétraitement du fichier
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            cleaned_content = self.text_processor.preprocess_text(content)
            complexity_stats = self.text_processor.analyze_text_complexity(cleaned_content)
            self.logger.info(f"Complexité du texte pour {filename}: {complexity_stats}")

            # Génération des chunks
            chunks_with_metadata = self.text_processor.split_into_chunks_with_header(cleaned_content, filename)
            
            chunks = []
            embeddings = []
            failures = 0

            # Traitement de chaque chunk
            for chunk_text, metadata in tqdm(chunks_with_metadata, desc=f"Processing {filename}"):
                embedding = self.embedding_generator.get_embedding(chunk_text)
                if embedding is not None:
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            'filename': metadata.filename,
                            'chunk_id': metadata.chunk_id,
                            'token_count': metadata.token_count,
                            'start_position': metadata.start_position,
                            'end_position': metadata.end_position
                        }
                    })
                    embeddings.append(embedding)
                else:
                    failures += 1
                    self.stats.add_error('embedding_generation_failed')

            # Mise à jour des statistiques
            duration = time.time() - start_time
            self.stats.add_file_stats(duration, len(chunks_with_metadata), failures)

            return chunks, embeddings

        except Exception as e:
            self.logger.error(f"Erreur lors du traitement de {filename}: {str(e)}")
            self.stats.add_error('file_processing_failed')
            return [], []

    def save_results(self, chunks: List[Dict[str, Any]], embeddings: List[np.ndarray]) -> None:
        try:
            # Sauvegarde des chunks au format JSON
            with open(self.config['output']['chunk_file'], 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

            # Sauvegarde des embeddings au format NPY
            embeddings_array = np.array(embeddings)
            np.save(self.config['output']['embedding_file'], embeddings_array)

            self.logger.info(f"Résultats sauvegardés avec succès")
            self.logger.info(f"Nombre total de chunks: {len(chunks)}")
            self.logger.info(f"Shape des embeddings: {embeddings_array.shape}")

        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde des résultats: {str(e)}")
            raise

    def run(self, folder_path: str) -> None:
        if not os.path.exists(folder_path):
            raise ValueError(f"Le dossier {folder_path} n'existe pas")

        files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        if not files:
            raise ValueError(f"Aucun fichier .txt trouvé dans {folder_path}")

        self.logger.info(f"Début du traitement de {len(files)} fichiers")
        all_chunks = []
        all_embeddings = []

        for filename in tqdm(files, desc="Processing files"):
            file_path = os.path.join(folder_path, filename)
            chunks, embeddings = self.process_file(file_path)
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)

        self.save_results(all_chunks, all_embeddings)
        
        # Affichage des statistiques finales
        summary = self.stats.get_summary()
        self.logger.info("Statistiques de traitement:")
        for key, value in summary.items():
            self.logger.info(f"{key}: {value}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Générateur d\'embeddings pour des fichiers texte')
    parser.add_argument('folder_path', help='Chemin vers le dossier contenant les fichiers texte')
    parser.add_argument('--config', default='config/settings.yaml', help='Chemin vers le fichier de configuration')
    args = parser.parse_args()

    try:
        pipeline = EmbeddingPipeline(args.config)
        pipeline.run(args.folder_path)
    except Exception as e:
        print(f"Erreur critique: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()

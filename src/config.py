"""Configuration management for Local GraphRAG."""
import yaml
from dataclasses import dataclass, field
from typing import List


# Single source of truth for the ablation matrix. A tag maps to the feature
# flags it enables; runners call apply_tag() so the recorded tag and the actual
# configuration can never drift apart (during development they once did).
TAG_FLAGS = {
    "base":        {"use_hyperedges": False, "compression": False},
    "ehrag":       {"use_hyperedges": True,  "compression": False},
    "compression": {"use_hyperedges": False, "compression": True},
    "full":        {"use_hyperedges": True,  "compression": True},
}


def apply_tag(config: "Config", tag: str) -> "Config":
    """Overwrite the ablation flags on `config` from the tag. Authoritative."""
    flags = TAG_FLAGS[tag]
    config.ehrag.use_hyperedges = flags["use_hyperedges"]
    config.compression.enabled = flags["compression"]
    return config


def model_slug(model_id: str) -> str:
    """e.g. 'mlx-community/Meta-Llama-3.1-8B-Instruct-4bit' -> 'llama-3.1-8b'."""
    return (model_id.split("/")[-1].lower()
            .replace("-instruct", "").replace("-4bit", "").replace("meta-", ""))


@dataclass
class HardwareConfig:
    chip: str = "M2"
    ram_gb: int = 16
    use_gpu: bool = True
    max_corpus_tokens: int = 5_000_000


@dataclass
class PathsConfig:
    data_dir: str = "./data"
    models_dir: str = "./models"
    results_dir: str = "./results"
    db_path: str = "./data/graph.duckdb"


@dataclass
class NLPConfig:
    spacy_model: str = "en_core_web_trf"
    max_entities_per_passage: int = 100


@dataclass
class EmbeddingConfig:
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32


@dataclass
class EHRAGConfig:
    use_hyperedges: bool = True
    birch_threshold: float = 0.5
    birch_branching_factor: int = 50
    semantic_top_k: int = 5
    decay_factor: float = 0.95
    # Query-time expansion shaping (2026-07-19 diagnosis: blanket expansion put
    # ~49% of PPR restart mass on cluster-mates and cut gold recall@20 by 10pts).
    # Legacy behavior = expand_seed_entities True, expansion_max_cluster 0 (off),
    # expansion_weight "flat", expansion_exclude_labels empty.
    expand_seed_entities: bool = True     # False: only exact NER matches expand
    expansion_max_cluster: int = 0        # >0: skip expansion sets larger than this
    expansion_weight: str = "flat"        # "flat": 0.5*w each | "inverse_size": 0.5*w/|set|
    expansion_exclude_labels: List[str] = field(default_factory=list)
    # e.g. ["DATE","CARDINAL","ORDINAL","QUANTITY","TIME","PERCENT","MONEY"] —
    # BIRCH lumps these into mega-clusters (762 dates in one) that flood PPR.


@dataclass
class VectorIndexConfig:
    index_type: str = "IVF_PQ"
    nlist: int = 100
    m: int = 16
    nbits: int = 8
    nprobe: int = 10


@dataclass
class PPRConfig:
    alpha: float = 0.15
    max_iterations: int = 50
    convergence_threshold: float = 1e-6
    top_k_passages: int = 10
    # When exact NER matches exist, rescale the vector-seed pool so seeds carry
    # at most this fraction of total restart mass. 0 = legacy (each seed keeps
    # 0.5*cosine, which lets a degree-1 name-lookalike seed concentrate more
    # mass on its single passage than exact query entities spread over theirs).
    seed_mass_fraction: float = 0.0


@dataclass
class CompressionConfig:
    enabled: bool = True
    method: str = "extractive_evidence"
    max_tokens: int = 2048
    preserve_spans: List[str] = field(default_factory=lambda: [
        "named_entities", "numerical_values", "noun_phrases"
    ])


@dataclass
class LLMConfig:
    model: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
    max_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    repetition_penalty: float = 1.1


@dataclass
class EvaluationConfig:
    benchmarks: List[str] = field(default_factory=lambda: ["hotpotqa"])
    max_eval_samples: int = 500


@dataclass
class Config:
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    ehrag: EHRAGConfig = field(default_factory=EHRAGConfig)
    vector_index: VectorIndexConfig = field(default_factory=VectorIndexConfig)
    ppr: PPRConfig = field(default_factory=PPRConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(
            hardware=HardwareConfig(**data.get("hardware", {})),
            paths=PathsConfig(**data.get("paths", {})),
            nlp=NLPConfig(**data.get("nlp", {})),
            embeddings=EmbeddingConfig(**data.get("embeddings", {})),
            ehrag=EHRAGConfig(**data.get("ehrag", {})),
            vector_index=VectorIndexConfig(**data.get("vector_index", {})),
            ppr=PPRConfig(**data.get("ppr", {})),
            compression=CompressionConfig(**data.get("compression", {})),
            llm=LLMConfig(**data.get("llm", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
        )

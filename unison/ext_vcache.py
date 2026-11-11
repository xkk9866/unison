"""Load the official vCache implementation (github.com/vcache-project/vCache, master) without its
cloud/LLM client dependencies. Only the decision policy (VerifiedDecisionPolicy), the cache, the
HNSW vector store and the metadata store are used; client-library modules are replaced by inert stubs
because our inference engine and embedding engine are passed in explicitly."""
import importlib.abc
import importlib.machinery
import os
import sys
import types
from unittest import mock

VCACHE_DIR = os.environ.get("VCACHE_DIR", "/home/yaokang/ext/vCache-master")
_STUB = ("langchain", "langchain_community", "langchain_openai", "langchain_anthropic", "langchain_huggingface",
         "langchain_google_genai", "openai", "chromadb", "anthropic")


_STUB_MOD = ("vcache.inference_engine.strategies.vllm",)  # imports vLLM>=0.10 APIs; we pass our own engine


class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in _STUB or name in _STUB_MOD:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = mock.MagicMock(name=spec.name)
        m.__path__, m.__spec__, m.__name__ = [], spec, spec.name
        return m

    def exec_module(self, module):
        pass


def load():
    if not any(isinstance(f, _Stub) for f in sys.meta_path):
        sys.meta_path.insert(0, _Stub())
    if VCACHE_DIR not in sys.path:
        sys.path.insert(0, VCACHE_DIR)
    from vcache.config import VCacheConfig
    from vcache.main import VCache
    from vcache.vcache_policy.strategies.verified import VerifiedDecisionPolicy
    from vcache.inference_engine.inference_engine import InferenceEngine
    from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine
    from vcache.vcache_core.cache.embedding_store.vector_db.strategies.hnsw_lib import HNSWLibVectorDB
    from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.strategies.in_memory import (
        InMemoryEmbeddingMetadataStorage)
    from vcache.vcache_core.similarity_evaluator.similarity_evaluator import SimilarityEvaluator
    return types.SimpleNamespace(**locals())

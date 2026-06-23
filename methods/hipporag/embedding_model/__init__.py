from .base import EmbeddingConfig, BaseEmbeddingModel

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    if "GritLM" in embedding_model_name:
        from .GritLM import GritLMEmbeddingModel

        return GritLMEmbeddingModel
    elif "NV-Embed-v2" in embedding_model_name:
        from .NVEmbedV2 import NVEmbedV2EmbeddingModel

        return NVEmbedV2EmbeddingModel
    elif "contriever" in embedding_model_name:
        from .Contriever import ContrieverModel

        return ContrieverModel
    else:
        from .OpenAIEmbedding import OpenAI_Compatible_EmbeddingModel

        logger.info(f"Unknown embedding model name: {embedding_model_name}, using OpenAI_Compatible_EmbeddingModel as default")
        return OpenAI_Compatible_EmbeddingModel

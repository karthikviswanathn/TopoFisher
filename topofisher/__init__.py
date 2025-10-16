"""
TopoFisher: Topological Fisher Information Analysis
"""
from .core.data_types import FisherConfig, FisherResult, TrainingConfig
from .core.pipeline import FisherPipeline
from .core.cached_pipeline import (
    CachedFisherPipeline,
    save_diagrams,
    load_diagrams,
    generate_and_save_diagrams
)
from .simulators.grf import GRFSimulator
from .simulators.gaussian_vector import GaussianVectorSimulator
from .filtrations.cubical import CubicalLayer
from .filtrations.mma import MMALayer
from .filtrations.identity import IdentityFiltration
from .vectorizations.topk import TopKLayer
from .vectorizations.combined import CombinedVectorization
from .vectorizations.persistence_image import PersistenceImageLayer
from .vectorizations.mma_topk import MMATopKLayer
from .vectorizations.identity import IdentityVectorization
from .compressions import (
    Compression,
    IdentityCompression,
    MOPEDCompression,
    MLPCompression,
    CNNCompression,
    InceptBlockCompression
)
from .fisher.analyzer import FisherAnalyzer
from .fisher.gaussianity import test_gaussianity

__version__ = "0.2.0"

__all__ = [
    "FisherConfig",
    "FisherResult",
    "TrainingConfig",
    "FisherPipeline",
    "CachedFisherPipeline",
    "save_diagrams",
    "load_diagrams",
    "generate_and_save_diagrams",
    "GRFSimulator",
    "GaussianVectorSimulator",
    "CubicalLayer",
    "MMALayer",
    "IdentityFiltration",
    "TopKLayer",
    "CombinedVectorization",
    "PersistenceImageLayer",
    "MMATopKLayer",
    "IdentityVectorization",
    "Compression",
    "IdentityCompression",
    "MOPEDCompression",
    "MLPCompression",
    "CNNCompression",
    "InceptBlockCompression",
    "FisherAnalyzer",
    "test_gaussianity",
]

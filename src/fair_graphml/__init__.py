"""Fair graph machine learning with a uniform estimator interface."""
from .inprocessing import (GAE, FairGNN, FairVGNN, NIFTY, UGE, DELTR, Sinkhorn,
    ThreeClassifiers, SinkhornMultiClass, ThreeClassifiersMultiClass, FairAdj, FairADJ, NCN, SEAL)
from .preprocessing import FairLP, EDITS, EDITSBatched, FairEGM, GFO, CFO, FEW, Graphair, GraphAir
from .postprocessing import DetConstSort, EpsilonGreedy
from .embedding import FairWalk, Node2Vec
from .heuristics import CommonNeighbors, AdamicAdar, PersonalizedPageRank

__version__ = "0.1.0"

METHODS = {
    "inprocessing": {cls.__name__: cls for cls in (FairGNN, FairVGNN, NIFTY, UGE, DELTR, FairAdj, Sinkhorn, ThreeClassifiers, SinkhornMultiClass, GAE, NCN, SEAL)},
    "preprocessing": {cls.__name__: cls for cls in (FairLP, EDITS, EDITSBatched, FairEGM, GFO, CFO, FEW, Graphair)},
    "postprocessing": {cls.__name__: cls for cls in (DetConstSort, EpsilonGreedy)},
    "embedding": {cls.__name__: cls for cls in (FairWalk, Node2Vec)},
    "heuristics": {cls.__name__: cls for cls in (CommonNeighbors, AdamicAdar, PersonalizedPageRank)},
}


def list_methods(category=None):
    """Discover available estimator names, grouped by category by default."""
    if category is not None:
        if category not in METHODS:
            raise ValueError(f"category must be one of {tuple(METHODS)}.")
        return sorted(METHODS[category])
    return {name: sorted(methods) for name, methods in METHODS.items()}


__all__ = [name for methods in METHODS.values() for name in methods] + ["FairADJ", "GraphAir", "ThreeClassifiersMultiClass", "list_methods"]

import itertools
from typing import Hashable, Iterable, Literal, Tuple, Set

def tqdm(iterable, **kwargs):
    return iterable

import networkx as nx

Edge = Tuple[Hashable, Hashable]


def fairlp(
    G: nx.Graph,
    protected_nodes: Iterable[Hashable],
    mode: Literal["avg", "min", "max"] = "avg",
) -> Tuple[nx.Graph, Set[Edge], Set[Edge]]:
    """
    Implementation of Algorithm 1 (FairLP: pre-processing fairness method for link prediction).

    Parameters
    ----------
    G : nx.Graph
        Undirected, unweighted simple graph. This graph will be modified in-place.
    protected_nodes : iterable
        Nodes belonging to the *protected node group* V* (e.g., minority).
        All remaining nodes in G are treated as unprotected.
    mode : {"avg", "min", "max"}, default "avg"
        How to “flatten” the densities:
            - "avg":  r = (density_prot + density_unprot)/2      (AvgD)
            - "min":  r = min(density_prot, density_unprot)      (MinD)
            - "max":  r = max(density_prot, density_unprot)      (MaxD)

    Returns
    -------
    G : nx.Graph
        The modified graph (same object as input, but with edges added/removed).
    E_plus : set of (u, v)
        Edges that were added.
    E_minus : set of (u, v)
        Edges that were removed.
    """

    # --- basic setup ---------------------------------------------------------
    if not isinstance(G, nx.Graph) or G.is_directed():
        raise ValueError("G must be an undirected simple graph (nx.Graph).")

    protected_nodes = set(protected_nodes)
    all_nodes = set(G.nodes())
    unprotected_nodes = all_nodes - protected_nodes

    if not protected_nodes or not unprotected_nodes:
        raise ValueError("Both protected and unprotected node groups must be non-empty.")

    def normalize_edge(u, v) -> Edge:
        return (u, v) if u <= v else (v, u)

    # Current edges, normalized (undirected)
    current_edges = {normalize_edge(u, v) for u, v in G.edges()}

    # --- define edge group membership ---------------------------------------
    # Protected edge group E*  = inter-group edges (one protected, one unprotected)
    # Unprotected edge group E¯ = intra-group edges (both protected or both unprotected)

    def is_protected_edge(u, v) -> bool:
        return (u in protected_nodes) ^ (v in protected_nodes)  # XOR

    # Build initial edge group sets from the current graph
    E_prot = set()
    E_unprot = set()
    for u, v in current_edges:
        if is_protected_edge(u, v):
            E_prot.add((u, v))
        else:
            E_unprot.add((u, v))

    # --- compute all possible edges (UEG) & cmax for each group -------------
    # Protected group: all possible inter-group pairs
    U_prot = {
        normalize_edge(u, v)
        for u in protected_nodes
        for v in unprotected_nodes
        if u != v
    }
    cmax_prot = len(U_prot)  # == |V_prot| * |V_unprot|

    # Unprotected group: all possible intra-group pairs within each node group
    U_unprot = set()
    # protected-protected pairs
    U_unprot.update(
        normalize_edge(u, v) for u, v in tqdm(itertools.combinations(protected_nodes, 2), desc='p-p pairs')
    )
    # unprotected-unprotected pairs
    U_unprot.update(
        normalize_edge(u, v) for u, v in tqdm(itertools.combinations(unprotected_nodes, 2), desc='up-up pairs')
    )
    cmax_unprot = len(U_unprot)  # == k1*(k1-1)/2 + k2*(k2-1)/2

    # --- densities of the two edge groups -----------------------------------
    def density(num_edges: int, cmax: int) -> float:
        return 0.0 if cmax == 0 else num_edges / cmax

    dens_prot = density(len(E_prot), cmax_prot)
    dens_unprot = density(len(E_unprot), cmax_unprot)

    if mode == "avg":
        r = 0.5 * (dens_prot + dens_unprot)
    elif mode == "min":
        r = min(dens_prot, dens_unprot)
    elif mode == "max":
        r = max(dens_prot, dens_unprot)
    else:
        raise ValueError("mode must be one of {'avg', 'min', 'max'}")

    # --- weight function: w(e) = |Γ(u) ∩ Γ(v)| ------------------------------
    def edge_weight(G_: nx.Graph, u, v) -> int:
        # number of common neighbors
        return len(set(G_.neighbors(u)).intersection(G_.neighbors(v)))

    # --- helper to adjust one group (protected or unprotected) --------------
    E_plus: Set[Edge] = set()
    E_minus: Set[Edge] = set()

    def adjust_group(
        G_: nx.Graph,
        E_group: Set[Edge],
        U_group: Set[Edge],
        cmax_group: int,
    ):
        nonlocal E_plus, E_minus

        creal = len(E_group)
        cexp = r * cmax_group

        # The paper defines k = cexp - creal; here we round to nearest integer.
        k = int(round(cexp - creal))

        if k == 0:
            return

        if k > 0:
            # Need to add k edges with *lowest* weight from NEG = U_group - E_group
            NEG = U_group - E_group
            for _ in tqdm(range(k), desc='adjusting group...'):
                if not NEG:
                    break
                # dynamic greedy: always pick lowest-weight edge w.r.t. current G_
                e = min(
                    NEG,
                    key=lambda e_: (edge_weight(G_, e_[0], e_[1]), e_[0], e_[1]),
                )
                NEG.remove(e)
                E_group.add(e)
                G_.add_edge(*e)
                E_plus.add(e)
        else:
            # Need to remove |k| edges with *lowest* weight from existing E_group
            k = -k
            for _ in tqdm(range(k), desc='adjusting group...'):
                if not E_group:
                    break
                e = min(
                    E_group,
                    key=lambda e_: (edge_weight(G_, e_[0], e_[1]), e_[0], e_[1]),
                )
                E_group.remove(e)
                if G_.has_edge(*e):
                    G_.remove_edge(*e)
                E_minus.add(e)

    # --- Step 2 in Algorithm 1: adjust each group ---------------------------
    adjust_group(G, E_prot, U_prot, cmax_prot)
    adjust_group(G, E_unprot, U_unprot, cmax_unprot)

    return G, E_plus, E_minus

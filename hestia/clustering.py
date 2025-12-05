import time
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from hestia.similarity import sim_df2mtx


def generate_clusters(
    df: pd.DataFrame,
    field_name: str,
    sim_df: pl.DataFrame,
    threshold: float = 0.4,
    verbose: int = 0,
    cluster_algorithm: str = 'greedy_incremental',
    filter_smaller: Optional[bool] = True,
    **kwargs
) -> np.ndarray:
    """Generates clusters from a DataFrame.

    This function supports several clustering algorithms that operate on
    pairwise similarity data. Each algorithm has different scalability,
    behavior, and underlying assumptions. Below is a summary of the available
    algorithms:

    Clustering algorithms:
        - `CDHIT` or `greedy_incremental`:
            Greedy incremental clustering similar to CD-HIT. Entities are
            sorted by length, and each new element seeds a cluster; all items
            above the similarity threshold are assigned to the same cluster.
            Fast, deterministic, and suitable for sequence-length-dependent
            ordering.

        - `greedy_cover_set` or `butina`:
            A greedy set-cover–style approach (similar to Butina clustering).
            Selects items with the largest number of neighbors above the
            threshold and forms clusters around them. Tends to produce compact,
            high-similarity groups.

        - `connected_components`:
            Treats similarity relations above the threshold as graph edges and
            computes connected components. All entities connected (directly or
            transitively) via similarity ≥ threshold belong to the same
            cluster. Very fast and stable for large sparse similarity graphs.

        - `bitbirch`:
            Clustering based on the BitBirch tree/hashing algorithm. Supports
            two modes:
                (1) fingerprint-based (e.g. SMILES → Morgan fingerprints), or
                (2) similarity-matrix-derived.
            Scales efficiently to large datasets and creates hierarchical,
            radius-based clusters.

        - `umap`:
            Reduces high-dimensional fingerprints or similarity matrices into a
            low-dimensional manifold using UMAP, then applies agglomerative
            clustering. Useful when clusters are better separated in embedded
            space than in raw feature or similarity space.

    :param df: DataFrame with entities to cluster.
    :type df: pd.DataFrame
    :param field_name: Name of the field with the entity information
    (e.g., `protein_sequence` or `structure_path`), defaults to 'sequence'.
    :type field_name: str
    :param threshold: Similarity value above which entities will be
    considered similar, defaults to 0.4
    :param sim_df: DataFrame with similarities (`metric`) between
    `query` and `target`, it is the product of `calculate_similarity` function
    :type sim_df: pl.DataFrame
    :type threshold: float
    :param verbose: How much information will be displayed.
    Options:
        - 0: Errors,
        - 1: Warnings,
        - 2: All
    Defaults to 0
    :type verbose: int
    :param cluster_algorithm: Clustering algorithm to use.
    Options:
        - `CDHIT` or `greedy_incremental`
        - `greedy_cover_set`
        - `connected_components`
        - `bitbirch`
        - `umap`

    Defaults to "greedy_incremental".
    :type cluster_algorithm: str, optional
    :param filter_smaller: Whether to filter smaller indices when constructing
    adjacency matrices in similarity-based algorithms, defaults to True.
    :type filter_smaller: bool, optional
    :raises NotImplementedError: Clustering algorithm is not supported
    :return: DataFrame with entities and the cluster they belong to.
    :rtype: np.ndarray
    """

    start = time.time()
    if isinstance(sim_df, pl.DataFrame):
        sim_df = sim_df.to_pandas()

    if cluster_algorithm in ['greedy_incremental', 'CDHIT']:
        clusters = _greedy_incremental_clustering(df, field_name, sim_df,
                                                  threshold, verbose)
    elif cluster_algorithm in ['greedy_cover_set', 'butina']:
        clusters = _greedy_cover_set(df, sim_df, threshold, verbose)
    elif cluster_algorithm in ['connected_components']:
        clusters = _connected_components_clustering(df, sim_df, threshold,
                                                    verbose, filter_smaller)
    elif cluster_algorithm in ['bitbirch']:
        clusters = _bitbirch_clustering(
            df,
            field_name=field_name,
            verbose=verbose,
            sim_df=sim_df,
            threshold=threshold,
            **kwargs
        )
    elif cluster_algorithm in ['umap']:
        clusters = _umap_clustering(
            df,
            field_name=field_name,
            verbose=verbose,
            threshold=threshold,
            sim_df=sim_df,
            filter_smaller=filter_smaller,
            **kwargs
        )
    else:
        raise NotImplementedError(
            f'Clustering algorithm: {cluster_algorithm} is not supported'
        )
    if verbose > 2:
        print(f'Clustering has taken {time.time() - start:.3f} s to compute.')

    return clusters


def _greedy_incremental_clustering(
    df: pd.DataFrame,
    field_name: str,
    sim_df: pd.DataFrame,
    threshold: float,
    verbose: int
) -> np.ndarray:
    df['lengths'] = df[field_name].map(len)
    df.sort_values(by='lengths', ascending=False, inplace=True)

    clusters = []
    clustered = set()
    sim_df = sim_df[sim_df['metric'] > threshold]

    if verbose > 2:
        pbar = tqdm(df.index)
    else:
        pbar = df.index

    for i in pbar:
        if i in clustered:
            continue
        in_cluster = set(sim_df.loc[sim_df['query'] == i, 'target'])
        in_cluster.update(set(sim_df.loc[sim_df['target'] == i, 'query']))
        in_cluster.update(set([i]))
        in_cluster = in_cluster.difference(clustered)

        for j in in_cluster:
            clusters.append(i)
        clustered.update(in_cluster)

    if verbose > 1:
        print('Clustering has generated:',
              f'{len(np.unique(clusters)):,d} clusters for',
              f'{len(df):,} entities')
    return np.array(clusters)


def _greedy_cover_set(
    df: pd.DataFrame,
    sim_df: pd.DataFrame,
    threshold: float,
    verbose: int
) -> np.ndarray:
    def _find_connectivity(df, sim_df):
        neighbours = []
        for i in df.index:
            in_cluster = set(sim_df.loc[sim_df['query'] == i, 'target'])
            in_cluster.update(set(sim_df.loc[sim_df['target'] == i, 'query']))
            neighbours.append(in_cluster)
        return neighbours
    sim_df = sim_df[sim_df['metric'] > threshold]
    neighbours = _find_connectivity(df, sim_df)
    order = np.argsort(neighbours)[::-1]

    clusters = np.zeros((len(df)))
    clustered = set()
    if verbose > 2:
        pbar = tqdm(order)
    else:
        pbar = order

    for i in pbar:
        if i in clustered:
            continue
        in_cluster = neighbours[i]
        in_cluster.update([i])
        in_cluster = in_cluster.difference(clustered)
        clustered.update(in_cluster)

        for j in in_cluster:
            clusters[j] = i

    unique_clusters, _ = np.unique(clusters, return_counts=True)

    if verbose > 1:
        print('Clustering has generated:',
              f'{len(unique_clusters):,d} clusters for',
              f'{len(df):,} entities')
    return clusters


def _connected_components_clustering(
    df: pd.DataFrame,
    sim_df: pd.DataFrame,
    threshold: float,
    verbose: int,
    filter_smaller: Optional[bool] = True
) -> np.ndarray:
    from scipy.sparse.csgraph import connected_components

    matrix = sim_df2mtx(sim_df, len(df), len(df),
                        threshold=threshold,
                        filter_smaller=filter_smaller)
    n, labels = connected_components(matrix, directed=False,
                                     return_labels=True)
    if verbose > 2:
        print('Clustering has generated:',
              f'{n:,d} connected components for',
              f'{len(df):,} entities')
    return labels


def _bitbirch_clustering(
    df: pd.DataFrame,
    field_name: str,
    verbose: int,
    sim_df: pd.DataFrame = None,
    threshold: float = None,
    branching_factor: int = 50,
    filter_smaller: bool = True,
    radius: int = 2,
    n_clusters: int = 20,
    bits: int = 1024,
) -> np.ndarray:
    try:
        import bitbirch.bitbirch as bb
    except ImportError as e:
        if verbose > 0:
            print(f"Error message: {e}")
        raise ImportError("This function requires BitBIRCH. `pip install git+https://github.com/mqcomplab/bitbirch")
    if sim_df is None:
        try:
            from rdkit import Chem
            from rdkit.Chem import rdFingerprintGenerator
        except ImportError:
            raise ImportError(
                "This function requires rdkit. `pip install rdkit`"
            )
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=bits
        )
        mol_list = [Chem.MolFromSmiles(x) for x in df[field_name]]
        fp_list = [fp_gen.GetFingerprintAsNumPy(x)
                   if x is not None else np.zeros((bits, ))
                   for x in mol_list]
        mtx = np.stack(fp_list)
        bb.set_merge(merge_criterion='radius')
        bitbirch = bb.BitBirch(
            branching_factor=branching_factor,
            threshold=threshold
        )
        bitbirch.fit(mtx)
        cluster_list = bitbirch.get_cluster_mol_ids()
        n_molecules = mtx.shape[0]
        cluster_labels = [0] * n_molecules
        for cluster_id, indices in enumerate(cluster_list):
            for idx in indices:
                cluster_labels[idx] = cluster_id
        return cluster_labels
    else:
        from sklearn.cluster import Birch
        mtx = sim_df2mtx(
            sim_df, threshold=0.1,
            filter_smaller=filter_smaller,
            boolean_out=True).todense()
        mtx = np.asarray(mtx)
        bb = Birch(
            threshold=threshold,
            branching_factor=branching_factor,
            n_clusters=n_clusters
        )
        return bb.fit_predict(mtx)


def _umap_clustering(
    df: pd.DataFrame,
    field_name: str,
    verbose: int,
    boolean_out: bool = True,
    threshold: float = None,
    sim_df: pd.DataFrame = None,
    n_clusters: int = 10,
    n_neighbors: int = 15,
    n_components: int = 2,
    n_pcs: int = 50,
    min_dist: float = 0.1,
    radius: int = 2,
    bits: int = 1024,
    filter_smaller: Optional[bool] = True
) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError(
            "This function requires umap. `pip install umap-learn`"
        )

    if sim_df is None:
        try:
            from rdkit import Chem
            from rdkit.Chem import rdFingerprintGenerator
        except ImportError:
            raise ImportError(
                "This function requires rdkit. `pip install rdkit`"
            )
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=bits
        )
        mol_list = [Chem.MolFromSmiles(x) for x in df[field_name]]
        fp_list = [fp_gen.GetFingerprintAsNumPy(x)
                   if x is not None else np.zeros((bits, ))
                   for x in mol_list]
        mtx = np.stack(fp_list)
    else:
        mtx = sim_df2mtx(
            sim_df, threshold=threshold,
            filter_smaller=filter_smaller,
            boolean_out=boolean_out)
    pca = PCA(n_components=n_pcs)
    pcs = pca.fit_transform(mtx)
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist
    )
    embedding = reducer.fit_transform(pcs)
    ac = AgglomerativeClustering(n_clusters=n_clusters)
    ac.fit_predict(embedding)
    if verbose > 2:
        print('Clustering has generated:',
              f'{len(np.unique(ac.labels_)):,d} clusters for',
              f'{len(df):,} entities')
    return ac.labels_

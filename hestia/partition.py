from multiprocessing import cpu_count
from typing import Tuple, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from hestia.similarity import sim_df2mtx
from hestia.clustering import generate_clusters
from hestia.reduction import similarity_reduction
from hestia.utils import (_assign_partitions, _cluster_reassignment,
                          _neighbour_analysis, _balanced_labels,
                          limited_agglomerative_clustering, _discretizer)


def smallest_assignment(clusters: np.ndarray, labels: np.ndarray,
                        size: int, valid_size: int,
                        test_size: int):
    """Assigns iteratively the smallest subclusters to the test subset, until
    it reaches the desired size.

    :param list_ids: Ordered list of item identifiers to assign.
    :type list_ids: list[str]
    :param partition_lengths: Desired number of items for each partition.
    :type partition_lengths: np.ndarray
    :param max_length_per_partition: Maximum allowed size for any partition,
        values in ``partition_lengths`` exceeding this are clipped, defaults to
        ``100000000``.
    :type max_length_per_partition: int, optional
    :return: The indices for training, testing and valiation subsets.
            In that order.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    # Initialize empty lists for train, test, and valid sets
    train, test, valid = [], [], []

    unique_parts, part_counts = np.unique(clusters, return_counts=True)
    sorted_parts = unique_parts[np.argsort(part_counts)]

    # Precompute indices for test and valid partitions
    for part in sorted_parts:
        part_indices = np.where(clusters == part)[0]

        if _balanced_labels(labels, part_indices, test, test_size, size):
            test.extend(part_indices)

    # Avoid test data points in valid set
    for part in sorted_parts:
        part_indices = np.where(clusters == part)[0]
        remaining_indices = [i for i in part_indices if i not in test]

        if remaining_indices:
            if _balanced_labels(labels, remaining_indices, valid, valid_size, size) and valid_size > 0:
                valid.extend(remaining_indices)
            else:
                train.extend(remaining_indices)
    return train, test, valid


def random_partition(
    df: pd.DataFrame,
    test_size: float,
    random_state: int = 42,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Use random partitioning algorithm
    to generate training and evaluation subsets.
    Wrapper around the `train_test_split` function
    from scikit-learn.

    :param df:  DataFrame with the entities to partition
    :type df: pd.DataFrame
    :param test_size: Proportion of entities to be allocated to
    test subset, defaults to 0.2
    :type test_size: float
    :param random_state: Seed for pseudo-random number
    generator algorithm, defaults to 42
    :type random_state: int, optional
    :return:  A tuple with the indexes of training and evaluation samples.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
    """
    train_df, test_df = train_test_split(df.index.tolist(),
                                         test_size=test_size,
                                         random_state=random_state)
    return train_df, test_df


def ccpart_random(
    df: pd.DataFrame,
    sim_df: pd.DataFrame,
    field_name: str = None,
    label_name: str = None,
    test_size: float = 0.2,
    valid_size: float = 0.0,
    threshold: float = 0.3,
    verbose: int = 0,
    seed: int = 0,
    n_bins: int = 10,
    filter_smaller: Optional[bool] = True
) -> Union[Tuple[list, list, list], Tuple[list, list, list, list]]:
    """
    Partitions a dataset into training, testing, and optional validation sets based on connected 
    component clustering using a similarity matrix. Ensures clusters are kept intact across splits 
    and optionally balances label distributions across partitions. Cluesters are assigned to 
    testing randomly.

    :param df: DataFrame containing the dataset to be partitioned.
    :type df: pd.DataFrame
    :param sim_df: DataFrame representing precomputed pairwise similarities between samples.
    :type sim_df: pd.DataFrame
    :param field_name: Name of the column in `df` used for clustering; if None, uses `sim_df` directly.
    :type field_name: str, optional
    :param label_name: Name of the label column for balancing partitions; if None, no balancing is performed.
    :type label_name: str, optional
    :param test_size: Fraction of the dataset to allocate to the test set.
    :type test_size: float
    :param valid_size: Fraction of the dataset to allocate to the validation set; set to 0.0 to skip validation split.
    :type valid_size: float
    :param threshold: Similarity threshold for connecting components when clustering.
    :type threshold: float
    :param verbose: Verbosity level for logging (higher values provide more detailed output).
    :type verbose: int
    :param n_bins: Number of bins to discretize continuous labels into for balancing purposes.
    :type n_bins: int
    :param filter_smaller: Whether with the similarity metric less is less similar.
    :type filter_smaller: bool, optional
    :return:
        - If `valid_size > 0`: returns (train_indices, test_indices, valid_indices, cluster_assignments)
        - Otherwise: returns (train_indices, test_indices, cluster_assignments)
    :rtype: Union[Tuple[list, list, list], Tuple[list, list, list, list]]
    """
    size = len(df)
    expected_test = test_size * size
    expected_valid = valid_size * size

    labels = df[label_name].to_numpy() if label_name else None
    labels = _discretizer(labels, n_bins=n_bins)

    # Generate cluster assignments
    clusters = generate_clusters(
        df,
        field_name=field_name,
        threshold=threshold,
        verbose=verbose,
        cluster_algorithm='connected_components',
        sim_df=sim_df,
        filter_smaller=filter_smaller
    )

    unique_parts, part_counts = np.unique(clusters, return_counts=True)
    # sorted_parts = unique_parts[np.argsort(part_counts)]
    np.random.seed(seed)
    np.random.shuffle(unique_parts)
    # Initialize empty lists for train, test, and valid sets
    test = []
    valid = []
    train = []

    # Precompute indices for test and valid partitions
    for part in unique_parts:
        part_indices = np.where(clusters == part)[0]

        if _balanced_labels(labels, part_indices, test, test_size, size):
            test.extend(part_indices)

    # Avoid test data points in valid set
    for part in unique_parts:
        part_indices = np.where(clusters == part)[0]
        remaining_indices = [i for i in part_indices if i not in test]

        if remaining_indices:
            if _balanced_labels(labels, remaining_indices, valid, valid_size, size) and valid_size > 0:
                valid.extend(remaining_indices)
            else:
                train.extend(remaining_indices)

    # Verbose output
    if verbose > 2:
        print(f'Proportion train: {(len(train) / size) * 100:.2f} %')
        print(f'Proportion test: {(len(test) / size) * 100:.2f} %')
        print(f'Proportion valid: {(len(valid) / size) * 100:.2f} %')

    # Warnings if the sizes of partitions are smaller than expected
    if len(test) < expected_test * 0.9 and verbose > 1:
        print(f'Warning: Proportion of test partition is smaller than expected: {(len(test) / size) * 100:.2f} %')
    if len(valid) < expected_valid * 0.9 and verbose > 1:
        print(f'Warning: Proportion of validation partition is smaller than expected: {(len(valid) / size) * 100:.2f} %')

    if valid_size > 0:
        return train, test, valid, clusters
    else:
        return train, test, clusters


def ccpart(
    df: pd.DataFrame,
    sim_df: pd.DataFrame,
    field_name: str = None,
    label_name: str = None,
    test_size: float = 0.2,
    valid_size: float = 0.0,
    threshold: float = 0.3,
    verbose: int = 0,
    n_bins: int = 10,
    filter_smaller: Optional[bool] = True
) -> Union[Tuple[list, list, list], Tuple[list, list, list, list]]:
    """
    Partitions a dataset into training, testing, and optional validation sets based on connected 
    component clustering using a similarity matrix. Ensures clusters are kept intact across splits 
    and optionally balances label distributions across partitions. Smallest clusters are iteratively
    assigned to testing.

    :param df: DataFrame containing the dataset to be partitioned.
    :type df: pd.DataFrame
    :param sim_df: DataFrame representing precomputed pairwise similarities between samples.
    :type sim_df: pd.DataFrame
    :param field_name: Name of the column in `df` used for clustering; if None, uses `sim_df` directly.
    :type field_name: str, optional
    :param label_name: Name of the label column for balancing partitions; if None, no balancing is performed.
    :type label_name: str, optional
    :param test_size: Fraction of the dataset to allocate to the test set.
    :type test_size: float
    :param valid_size: Fraction of the dataset to allocate to the validation set; set to 0.0 to skip validation split.
    :type valid_size: float
    :param threshold: Similarity threshold for connecting components when clustering.
    :type threshold: float
    :param verbose: Verbosity level for logging (higher values provide more detailed output).
    :type verbose: int
    :param n_bins: Number of bins to discretize continuous labels into for balancing purposes.
    :type n_bins: int
    :param filter_smaller: Whether with the similarity metric less is less similar.
    :type filter_smaller: bool, optional

    :return:
        - If `valid_size > 0`: returns (train_indices, test_indices, valid_indices, cluster_assignments)
        - Otherwise: returns (train_indices, test_indices, cluster_assignments)
    :rtype: Union[Tuple[list, list, list], Tuple[list, list, list, list]]
    """
    size = len(df)
    expected_test = test_size * size
    expected_valid = valid_size * size

    labels = df[label_name].to_numpy() if label_name else None
    labels = _discretizer(labels, n_bins=n_bins)

    # Generate cluster assignments
    clusters = generate_clusters(
        df,
        field_name=field_name,
        threshold=threshold,
        verbose=verbose,
        cluster_algorithm='connected_components',
        sim_df=sim_df,
        filter_smaller=filter_smaller
    )
    train, test, valid = smallest_assignment(
        clusters, labels, size,
        valid_size, test_size
    )

    # Verbose output
    if verbose > 2:
        print(f'Proportion train: {(len(train) / size) * 100:.2f} %')
        print(f'Proportion test: {(len(test) / size) * 100:.2f} %')
        print(f'Proportion valid: {(len(valid) / size) * 100:.2f} %')

    # Warnings if the sizes of partitions are smaller than expected
    if len(test) < expected_test * 0.9 and verbose > 1:
        print(f'Warning: Proportion of test partition is smaller than expected: {(len(test) / size) * 100:.2f} %')
    if len(valid) < expected_valid * 0.9 and verbose > 1:
        print(f'Warning: Proportion of validation partition is smaller than expected: {(len(valid) / size) * 100:.2f} %')

    if valid_size > 0:
        return train, test, valid, clusters
    else:
        return train, test, clusters


def cdhit_part(
    df: pd.DataFrame,
    sim_df: pd.DataFrame,
    field_name: str = None,
    label_name: str = None,
    test_size: float = 0.2,
    valid_size: float = 0.0,
    threshold: float = 0.3,
    verbose: int = 0,
    n_bins: int = 10,
    filter_smaller: Optional[bool] = True
) -> Union[Tuple[list, list, list], Tuple[list, list, list, list]]:
    """
    Partitions a dataset into training, testing, and optional validation sets based on connected 
    component clustering using a similarity matrix. Ensures clusters are kept intact across splits 
    and optionally balances label distributions across partitions. Smallest clusters are iteratively
    assigned to testing.

    :param df: DataFrame containing the dataset to be partitioned.
    :type df: pd.DataFrame
    :param sim_df: DataFrame representing precomputed pairwise similarities between samples.
    :type sim_df: pd.DataFrame
    :param field_name: Name of the column in `df` used for clustering; if None, uses `sim_df` directly.
    :type field_name: str, optional
    :param label_name: Name of the label column for balancing partitions; if None, no balancing is performed.
    :type label_name: str, optional
    :param test_size: Fraction of the dataset to allocate to the test set.
    :type test_size: float
    :param valid_size: Fraction of the dataset to allocate to the validation set; set to 0.0 to skip validation split.
    :type valid_size: float
    :param threshold: Similarity threshold for connecting components when clustering.
    :type threshold: float
    :param verbose: Verbosity level for logging (higher values provide more detailed output).
    :type verbose: int
    :param n_bins: Number of bins to discretize continuous labels into for balancing purposes.
    :type n_bins: int
    :param filter_smaller: Whether with the similarity metric less is less similar.
    :type filter_smaller: bool, optional

    :return:
        - If `valid_size > 0`: returns (train_indices, test_indices, valid_indices, cluster_assignments)
        - Otherwise: returns (train_indices, test_indices, cluster_assignments)
    :rtype: Union[Tuple[list, list, list], Tuple[list, list, list, list]]
    """
    size = len(df)
    expected_test = test_size * size
    expected_valid = valid_size * size

    labels = df[label_name].to_numpy() if label_name else None
    labels = _discretizer(labels, n_bins=n_bins)

    # Generate cluster assignments
    clusters = generate_clusters(
        df,
        field_name=field_name,
        threshold=threshold,
        verbose=verbose,
        cluster_algorithm='CDHIT',
        sim_df=sim_df,
        filter_smaller=filter_smaller
    )
    train, test, valid = smallest_assignment(
        clusters, labels, size,
        valid_size, test_size
    )

    # Verbose output
    if verbose > 2:
        print(f'Proportion train: {(len(train) / size) * 100:.2f} %')
        print(f'Proportion test: {(len(test) / size) * 100:.2f} %')
        print(f'Proportion valid: {(len(valid) / size) * 100:.2f} %')

    # Warnings if the sizes of partitions are smaller than expected
    if len(test) < expected_test * 0.9 and verbose > 1:
        print(f'Warning: Proportion of test partition is smaller than expected: {(len(test) / size) * 100:.2f} %')
    if len(valid) < expected_valid * 0.9 and verbose > 1:
        print(f'Warning: Proportion of validation partition is smaller than expected: {(len(valid) / size) * 100:.2f} %')

    if valid_size > 0:
        return train, test, valid, clusters
    else:
        return train, test, clusters


def reduction_partition(
    df: pd.DataFrame,
    sim_df: pd.DataFrame,
    field_name: str,
    sim_function: str = 'mmseqs+prefilter',
    threads: int = cpu_count(),
    clustering_mode: str = "CDHIT",
    denominator: str = "shortest",
    test_size: float = 0.2,
    valid_size: float = 0.0,
    threshold: float = 0.3,
    verbose: int = 2,
    data_type: str = 'protein',
    representation: str = '3di+aa',
    random_state: int = 42,
    bits: int = 1024,
    radius: int = 2,
    config: dict = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = similarity_reduction(df, sim_function, field_name,
                              threads, clustering_mode, denominator,
                              test_size, threshold, verbose, data_type,
                              representation, bits,
                              radius, sim_df, config)
    train, test = random_partition(df.index.tolist(), test_size=test_size,
                                   random_state=random_state)
    if valid_size > 0:
        adjust_valid = valid_size / (1 - test_size)
        train, valid = random_partition(train, test_size=adjust_valid,
                                        random_state=random_state)
        return train, test, valid
    else:
        return train, test


def graph_part(
    df: pd.DataFrame,
    sim_df: pd.DataFrame,
    label_name: str = None,
    test_size: float = 0.0,
    valid_size: float = 0.0,
    threshold: float = 0.3,
    verbose: int = 2,
    n_parts: int = 10,
    filter_smaller: Optional[bool] = True
):
    """
    Builds a graph from the provided similarity matrix, applies a limited
    agglomerative clustering algorithm, balances clusters across partitions,
    and performs iterative reassignment to minimize forbidden edges. The final
    output can optionally be split into train/test/validation subsets based on
    cluster proportions.

    Reference: Teufel F, Gíslason MH, Almagro Armenteros JJ, Johansen AR,
    Winther O, Nielsen H.
    GraphPart: homology partitioning for biological sequence analysis.
    NAR genomics and bioinformatics. 2023 Dec 1;5(4):lqad088.

    Code adapted and generalized from the project Github repository:
    https://github.com/graph-part/graph-part

    :param df: DataFrame containing the entities to partition.
    :type df: pd.DataFrame
    :param sim_df: Pairwise similarity DataFrame used to build the graph.
    :type sim_df: pd.DataFrame
    :param label_name: Optional column name containing entity labels used to
        guide cluster assignment for balancing, defaults to ``None``.
    :type label_name: str, optional
    :param test_size: Proportion of entities to allocate to the test split,
        defaults to ``0.0``.
    :type test_size: float, optional
    :param valid_size: Proportion of entities to allocate to the validation split
        (applied only after test split assignment), defaults to ``0.0``.
    :type valid_size: float, optional
    :param threshold: Similarity threshold used to define edges in the graph and
        guide clustering, defaults to ``0.3``.
    :type threshold: float, optional
    :param verbose: Verbosity level. Values above 1 enable progress information,
        defaults to ``2``.
    :type verbose: int, optional
    :param n_parts: Number of partitions (clusters) to generate, defaults to ``10``.
    :type n_parts: int, optional
    :param filter_smaller: If ``True``, edges with similarity >= threshold are kept.
        If ``False``, edges <= threshold are kept instead, defaults to ``True``.
    :type filter_smaller: bool, optional

    :return: If ``test_size`` and ``valid_size`` are both ``0.0``, returns an array
        of partition assignments (with ``-1`` for removed nodes). Otherwise returns
        train/test or train/test/valid index lists along with full cluster labels.
    :rtype: Union[
        np.ndarray,
        Tuple[List[int], List[int], np.ndarray],
        Tuple[List[int], List[int], List[int], np.ndarray]
    ]
    
    """
    mtx = sim_df2mtx(sim_df, len(df), boolean_out=False)
    if filter_smaller:
        mtx = mtx >= threshold
    else:
        mtx = mtx <= threshold

    if label_name is not None:
        labels = df[label_name]
    else:
        labels = np.zeros(mtx.shape[0], dtype=np.int8)
    if verbose > 1:
        print('Clustering using limited agglomerative clustering algorithm...')
    if n_parts is None:
        n_parts = 10
    clusters = limited_agglomerative_clustering(mtx, n_parts, threshold,
                                                labels, verbose=verbose)
    cluster_inds, cluster_sizes = np.unique(clusters, return_counts=True)
    unique_labs, lab_counts = np.unique(labels, return_counts=True)
    n_labels = len(unique_labs)
    cluster_labs = np.ones((n_parts, n_labels), dtype=int)

    if verbose > 1:
        print(f'Clustering generated {len(cluster_inds):,} clusters...')

    for ind in cluster_inds:
        clst_members = clusters == ind
        clst_labels = labels[clst_members]
        label, count_labels = np.unique(clst_labels, return_counts=True)
        clst_lab_count = cluster_labs.copy()
        clst_lab_count[:, label] += count_labels
        clst_lab_prop = cluster_labs / clst_lab_count
        best_group = np.argmin(np.sum(clst_lab_prop, axis=1))
        cluster_labs[best_group, label] += count_labels
        clusters[clst_members] = best_group

    cluster_labs = np.unique(clusters)

    mtx = mtx > threshold
    removed = np.ones(mtx.shape[0], dtype=np.int8) == 1
    clusters = _assign_partitions(clusters, labels, n_parts, verbose)
    mtx = mtx > threshold
    removed = np.ones(mtx.shape[0], dtype=np.int8) == 1
    i = 0
    if verbose > 1:
        pbar = tqdm()
    E_f = _neighbour_analysis(mtx, clusters)
    clus_labs, clusters_sizes = np.unique(clusters[removed],
                                          return_counts=True)
    if E_f.sum() == 0:
        re_clusters = clusters

    while E_f.sum() > 0:
        re_clusters, E_f = _cluster_reassignment(mtx, clusters, removed)
        i += 1

        if E_f.sum() > 0:
            num_to_remove = int(E_f.sum() * np.log10(i) / 100) + 1
            connectivity_inds = np.argsort(E_f)[-num_to_remove:]
            removed[connectivity_inds] = False
            if verbose > 1:
                mssg = f'Forbidden edges: {E_f.sum()} - Removed: '
                mssg += f'{mtx.shape[0] - removed.sum():,}'
                if verbose > 1:
                    pbar.set_description(mssg)
                    pbar.update(1)

        clus_labs, clusters_sizes = np.unique(re_clusters[removed],
                                              return_counts=True)
        if len(clus_labs) < n_parts:
            mssg = 'Dataset cannot be partitioned at current threshold '
            mssg += f'into {n_parts} partitions. '
            mssg += 'It leads to loss of a complete partition'
            raise RuntimeError(mssg)
    if verbose > 1:
        pbar.close()
        mssg = f'Number of entities removed: {mtx.shape[0] - removed.sum():,}'
        mssg += f' out of {mtx.shape[0]}'
        print(mssg)

    o_train, o_test, o_valid = [], [], []
    test_len, valid_len = 0, 0

    if test_size > 0.0:
        train, test = [], []
        for clus in clus_labs:
            members = re_clusters == clus
            cluster_size = members[removed].sum()

            if (cluster_size + test_len) / removed.sum() > test_size:
                train.append(clus)
            else:
                test_len += cluster_size
                test.append(clus)

        if valid_size > 0.0:
            new_train, valid = [], []
            for clus in train:
                members = re_clusters == clus
                cluster_size = members[removed].sum()

                if (cluster_size + valid_len) / removed.sum() > valid_size:
                    new_train.append(clus)
                else:
                    valid_len += cluster_size
                    valid.append(clus)

            for clus in new_train:
                members = np.argwhere((re_clusters == clus) * removed)
                for member in members:
                    o_train.append(member.tolist()[0])
            for clus in test:
                members = np.argwhere((re_clusters == clus) * removed)
                for member in members:
                    o_test.append(member.tolist()[0])
            for clus in valid:
                members = np.argwhere((re_clusters == clus) * removed)
                for member in members:
                    o_valid.append(member.tolist()[0])
            if verbose > 0:
                print('Proportion train:',
                    f'{(len(o_train) / removed.sum()) * 100:.2f} %')
                print('Proportion test:',
                    f'{(len(o_test) / removed.sum()) * 100:.2f} %')
                print('Proportion valid:',
                    f'{(len(o_valid) /  removed.sum()) * 100:.2f} %')
            return o_train, o_test, o_valid, clusters
        else:
            for clus in train:
                members = np.argwhere((re_clusters == clus) * removed)
                for member in members:
                    o_train.append(member.tolist()[0])
            for clus in test:
                members = np.argwhere((re_clusters == clus) * removed)
                for member in members:
                    o_test.append(member.tolist()[0])
            if verbose > 0:
                print('Proportion train:',
                    f'{(len(o_train) / removed.sum()) * 100:.2f} %')
                print('Proportion test:',
                    f'{(len(o_test) /  removed.sum()) * 100:.2f} %')
            return o_train, o_test, clusters

    re_clusters[~removed] = -1
    return re_clusters


def umap_original(
    df: pd.DataFrame,
    field_name: str,
    label_name: str = None,
    test_size: float = 0.0,
    valid_size: float = 0.0,
    threshold: float = 0.3,
    verbose: int = 2,
    n_clusters: int = 10,
    n_neighbors: int = 15,
    n_components: int = 2,
    n_pcs: int = 50,
    min_dist: float = 0.1,
    radius: int = 2,
    bits: int = 1024,
    n_bins: int = 10,
    **kwargs
):
    """Computes UMAP embeddings using the specified feature column, generates cluster
    assignments, discretizes labels (if provided), and then distributes the
    instances into train/test/validation partitions using the
    ``smallest_assignment`` strategy. Optional warnings are printed if resulting
    partitions deviate significantly from expected proportions.

    Reference: Guo Q, Hernandez-Hernandez S, Ballester PJ. UMAP-based clustering split for
    rigorous evaluation of AI models for virtual screening on cancer cell lines. Journal of Cheminformatics.
    2025 Jun 10;17(1):94.

    Code adapted from Pat Walter's useful rdkit utils Github Repository:
    https://github.com/PatWalters/useful_rdkit_utils

    :param df: Input DataFrame containing entities to cluster and partition.
    :type df: pd.DataFrame
    :param field_name: Name of the column containing the features used by UMAP.
    :type field_name: str
    :param label_name: Optional column name with labels used for balancing
        partitions, defaults to ``None``.
    :type label_name: str, optional
    :param test_size: Proportion of entities to place in the test subset,
        defaults to ``0.0``.
    :type test_size: float, optional
    :param valid_size: Proportion of entities to place in the validation subset,
        defaults to ``0.0``.
    :type valid_size: float, optional
    :param threshold: Threshold applied during UMAP-based graph clustering,
        defaults to ``0.3``.
    :type threshold: float, optional
    :param verbose: Verbosity level. Values ``> 2`` print partition proportions,
        defaults to ``2``.
    :type verbose: int, optional
    :param n_clusters: Desired number of clusters to generate using UMAP,
        defaults to ``10``.
    :type n_clusters: int, optional
    :param n_neighbors: UMAP ``n_neighbors`` parameter, defaults to ``15``.
    :type n_neighbors: int, optional
    :param n_components: Number of UMAP embedding dimensions, defaults to ``2``.
    :type n_components: int, optional
    :param n_pcs: Number of principal components to compute before UMAP,
        defaults to ``50``.
    :type n_pcs: int, optional
    :param min_dist: UMAP ``min_dist`` parameter controlling embedding tightness,
        defaults to ``0.1``.
    :type min_dist: float, optional
    :param radius: Radius value used by the UMAP graph construction, defaults to ``2``.
    :type radius: int, optional
    :param bits: Dimensionality of any hashing step used for vector representations,
        defaults to ``1024``.
    :type bits: int, optional
    :param n_bins: Number of bins used when discretizing labels for balancing,
        defaults to ``10``.
    :type n_bins: int, optional
    :param kwargs: Additional keyword arguments passed to underlying UMAP or
        clustering routines.
    :type kwargs: dict

    :return: If ``valid_size > 0`` returns train, test, valid partitions plus
        cluster assignments. Otherwise returns train, test partitions plus cluster
        assignments.
    :rtype: Union[
        Tuple[List[int], List[int], List[int], np.ndarray],
        Tuple[List[int], List[int], np.ndarray]
    ]
    """
    size = len(df)
    expected_test = test_size * size
    expected_valid = valid_size * size

    labels = df[label_name].to_numpy() if label_name else None
    labels = _discretizer(labels, n_bins=n_bins)

    # Generate cluster assignments
    clusters = generate_clusters(
        df,
        sim_df=None,
        field_name=field_name,
        threshold=threshold,
        verbose=verbose,
        cluster_algorithm='umap',
        n_clusters=n_clusters,
        n_neighbors=n_neighbors,
        n_components=n_components,
        n_pcs=n_pcs,
        min_dist=min_dist,
        radius=radius,
        bits=bits
    )
    train, test, valid = smallest_assignment(
        clusters, labels, size,
        valid_size, test_size
    )
    # Verbose output
    if verbose > 2:
        print(f'Proportion train: {(len(train) / size) * 100:.2f} %')
        print(f'Proportion test: {(len(test) / size) * 100:.2f} %')
        print(f'Proportion valid: {(len(valid) / size) * 100:.2f} %')

    # Warnings if the sizes of partitions are smaller than expected
    if len(test) < expected_test * 0.9 and verbose > 1:
        print(f'Warning: Proportion of test partition is smaller than expected: {(len(test) / size) * 100:.2f} %')
    if len(valid) < expected_valid * 0.9 and verbose > 1:
        print(f'Warning: Proportion of validation partition is smaller than expected: {(len(valid) / size) * 100:.2f} %')

    if valid_size > 0:
        return train, test, valid, clusters
    else:
        return train, test, clusters


def sim_umap(
    df: pd.DataFrame,
    sim_df: pl.DataFrame,
    field_name: str = None,
    label_name: str = None,
    test_size: float = 0.0,
    valid_size: float = 0.0,
    threshold: float = 0.3,
    verbose: int = 2,
    n_clusters: int = 10,
    n_neighbors: int = 15,
    n_components: int = 2,
    n_pcs: int = 50,
    min_dist: float = 0.1,
    boolean_out: bool = True,
    n_bins: int = 10,
):
    """UMAP-based partitioning using an external similarity matrix.

    It's a generalization of the `UMAP_original` algorithm from Guo et al., 2025,
    but extended to work on similarity matrices, instead of binary fingerprints.

    Generates clusters using UMAP while incorporating an external similarity
    matrix (``sim_df``). Optionally discretizes labels for balancing and partitions
    the dataset using ``smallest_assignment`` into train/test/validation subsets.
    Prints warnings if achieved proportions deviate significantly from expectations.

    :param df: Input DataFrame containing the entities to cluster and partition.
    :type df: pd.DataFrame
    :param sim_df: Similarity matrix provided as a Polars DataFrame, used to
        augment UMAP clustering.
    :type sim_df: pl.DataFrame
    :param field_name: Optional name of a column with feature vectors used by UMAP.
        If ``None``, clustering relies entirely on ``sim_df``, defaults to ``None``.
    :type field_name: str, optional
    :param label_name: Optional column name containing labels used for balancing
        partitions, defaults to ``None``.
    :type label_name: str, optional
    :param test_size: Proportion of entities to allocate to the test subset,
        defaults to ``0.0``.
    :type test_size: float, optional
    :param valid_size: Proportion of entities to allocate to the validation subset,
        defaults to ``0.0``.
    :type valid_size: float, optional
    :param threshold: Threshold used by the UMAP graph clustering step,
        defaults to ``0.3``.
    :type threshold: float, optional
    :param verbose: Verbosity level. When ``> 2`` prints detailed proportions,
        defaults to ``2``.
    :type verbose: int, optional
    :param n_clusters: Desired number of clusters to generate, defaults to ``10``.
    :type n_clusters: int, optional
    :param n_neighbors: UMAP ``n_neighbors`` parameter, defaults to ``15``.
    :type n_neighbors: int, optional
    :param n_components: Number of UMAP embedding dimensions, defaults to ``2``.
    :type n_components: int, optional
    :param n_pcs: Number of principal components to compute before UMAP,
        defaults to ``50``.
    :type n_pcs: int, optional
    :param min_dist: UMAP ``min_dist`` parameter controlling embedding tightness,
        defaults to ``0.1``.
    :type min_dist: float, optional
    :param boolean_out: Whether to convert the similarity thresholding output
        to boolean values, defaults to ``True``.
    :type boolean_out: bool, optional
    :param n_bins: Number of bins used when discretizing labels for balancing,
        defaults to ``10``.
    :type n_bins: int, optional

    :return: If ``valid_size > 0`` returns train, test, valid subsets plus
        cluster assignments. Otherwise returns train, test subsets plus
        cluster assignments.
    :rtype: Union[
        Tuple[List[int], List[int], List[int], np.ndarray],
        Tuple[List[int], List[int], np.ndarray]
    ]
    """
    size = len(df)
    expected_test = test_size * size
    expected_valid = valid_size * size

    labels = df[label_name].to_numpy() if label_name else None
    labels = _discretizer(labels, n_bins=n_bins)

    # Generate cluster assignments
    clusters = generate_clusters(
        df,
        sim_df=sim_df,
        field_name=field_name,
        threshold=threshold,
        verbose=verbose,
        cluster_algorithm='umap',
        n_clusters=n_clusters,
        n_neighbors=n_neighbors,
        n_components=n_components,
        boolean_out=boolean_out,
        n_pcs=n_pcs,
        min_dist=min_dist,
    )
    train, test, valid = smallest_assignment(
        clusters, labels, size,
        valid_size, test_size
    )
    # Verbose output
    if verbose > 2:
        print(f'Proportion train: {(len(train) / size) * 100:.2f} %')
        print(f'Proportion test: {(len(test) / size) * 100:.2f} %')
        print(f'Proportion valid: {(len(valid) / size) * 100:.2f} %')

    # Warnings if the sizes of partitions are smaller than expected
    if len(test) < expected_test * 0.9 and verbose > 1:
        print(f'Warning: Proportion of test partition is smaller than expected: {(len(test) / size) * 100:.2f} %')
    if len(valid) < expected_valid * 0.9 and verbose > 1:
        print(f'Warning: Proportion of validation partition is smaller than expected: {(len(valid) / size) * 100:.2f} %')

    if valid_size > 0:
        return train, test, valid, clusters
    else:
        return train, test, clusters


def scaffold(
    df: pd.DataFrame,
    field_name: str,
    label_name: str = None,
    test_size: float = 0.0,
    valid_size: float = 0.0,
    n_bins: int = 10,
    verbose: int = 1
):
    """Partition a dataset based on Bemis-Murcko scaffolds.

    Generates Bemis-Murcko scaffolds from the molecular SMILES in ``field_name``
    and assigns clusters based on unique scaffolds. Optionally discretizes labels
    for balancing and partitions the dataset into train/test/validation subsets
    using ``smallest_assignment``. Prints warnings if partition sizes deviate
    significantly from expectations.

    :param df: DataFrame containing molecular data.
    :type df: pd.DataFrame
    :param field_name: Column name containing SMILES strings used to generate scaffolds.
    :type field_name: str
    :param label_name: Optional column name containing labels used for balancing,
        defaults to ``None``.
    :type label_name: str, optional
    :param test_size: Proportion of entities to allocate to the test subset,
        defaults to ``0.0``.
    :type test_size: float, optional
    :param valid_size: Proportion of entities to allocate to the validation subset,
        defaults to ``0.0``.
    :type valid_size: float, optional
    :param n_bins: Number of bins used when discretizing labels for balancing,
        defaults to ``10``.
    :type n_bins: int, optional
    :param verbose: Verbosity level. When ``> 2`` prints detailed proportions,
        defaults to ``1``.
    :type verbose: int, optional

    :return: If ``valid_size > 0`` returns train, test, valid subsets plus cluster assignments.
        Otherwise returns train, test subsets plus cluster assignments.
    :rtype: Union[
        Tuple[List[int], List[int], List[int], np.ndarray],
        Tuple[List[int], List[int], np.ndarray]
    ]
    """
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    def _get_scaffold(smi: str) -> str:
        """
        Generate the Bemis-Murcko scaffold for a given molecule.

        :param smi: A SMILES string or an RDKit molecule object representing the
                    molecule for which to generate the scaffold.
        :return: A SMILES string representing the Bemis-Murcko scaffold of the input
                molecule. If the scaffold cannot be generated, the input SMILES
                string is returned.
        """
        try:
            scaffold = MurckoScaffoldSmiles(smi)
        except ValueError:
            scaffold = smi
        if len(scaffold) == 0:
            scaffold = smi
        return scaffold

    size = len(df)
    expected_test = test_size * size
    expected_valid = valid_size * size

    labels = df[label_name].to_numpy() if label_name else None
    labels = _discretizer(labels, n_bins=n_bins)

    scaffold_series = pd.Series([_get_scaffold(x)
                                 for x in df[field_name]])
    clusters, _ = pd.factorize(scaffold_series)
    train, test, valid = smallest_assignment(
        clusters, labels, size,
        valid_size, test_size
    )
    # Verbose output
    if verbose > 2:
        print(f'Proportion train: {(len(train) / size) * 100:.2f} %')
        print(f'Proportion test: {(len(test) / size) * 100:.2f} %')
        print(f'Proportion valid: {(len(valid) / size) * 100:.2f} %')

    # Warnings if the sizes of partitions are smaller than expected
    if len(test) < expected_test * 0.9 and verbose > 1:
        print(f'Warning: Proportion of test partition is smaller than expected: {(len(test) / size) * 100:.2f} %')
    if len(valid) < expected_valid * 0.9 and verbose > 1:
        print(f'Warning: Proportion of validation partition is smaller than expected: {(len(valid) / size) * 100:.2f} %')

    if valid_size > 0:
        return train, test, valid, clusters
    else:
        return train, test, clusters


def butina(
    df: pd.DataFrame,
    sim_df: pd.DataFrame,
    field_name: str = None,
    label_name: str = None,
    test_size: float = 0.2,
    valid_size: float = 0.0,
    threshold: float = 0.3,
    verbose: int = 0,
    n_bins: int = 10,
    filter_smaller: Optional[bool] = True
):
    """Partition a dataset using a Butina-style greedy clustering algorithm.

    Generates clusters based on molecular similarity using a greedy
    cover set approach (Butina clustering). Labels can be optionally
    discretized for balancing, and the dataset is partitioned into
    train/test/validation subsets using ``smallest_assignment``. Prints
    warnings if the partition sizes deviate from expectations.

    Generalized to work on similarity matrices rather than fingerprints directly.

    Reference: Butina D. Unsupervised data base clustering based on daylight's
    fingerprint and Tanimoto similarity: A fast and automated way
    to cluster small and large data sets.
    Journal of Chemical Information and Computer Sciences.
    1999 Jul 26;39(4):747-50.

    :param df: DataFrame containing molecular entities.
    :type df: pd.DataFrame
    :param sim_df: DataFrame or matrix containing pairwise similarity scores
        between entities.
    :type sim_df: pd.DataFrame
    :param field_name: Optional column name used for clustering. If None,
        clustering is based solely on ``sim_df``.
    :type field_name: str, optional
    :param label_name: Optional column name for labels used for balancing,
        defaults to ``None``.
    :type label_name: str, optional
    :param test_size: Proportion of entities to allocate to the test subset,
        defaults to 0.2.
    :type test_size: float, optional
    :param valid_size: Proportion of entities to allocate to the validation
        subset, defaults to 0.0.
    :type valid_size: float, optional
    :param threshold: Similarity threshold used for cluster formation,
        defaults to 0.3.
    :type threshold: float, optional
    :param verbose: Verbosity level. Higher values print detailed partition
        proportions, defaults to 0.
    :type verbose: int, optional
    :param n_bins: Number of bins used when discretizing labels for balancing,
        defaults to 10.
    :type n_bins: int, optional
    :param filter_smaller: Whether to filter smaller similarity values during
        clustering, defaults to True.
    :type filter_smaller: bool, optional

    :return: If ``valid_size > 0`` returns train, test, valid subsets plus cluster assignments.
        Otherwise returns train, test subsets plus cluster assignments.
    :rtype: Union[
        Tuple[List[int], List[int], List[int], np.ndarray],
        Tuple[List[int], List[int], np.ndarray]
    ]
    """
    size = len(df)
    expected_test = test_size * size
    expected_valid = valid_size * size

    labels = df[label_name].to_numpy() if label_name else None
    labels = _discretizer(labels, n_bins=n_bins)

    # Generate cluster assignments
    clusters = generate_clusters(
        df,
        field_name=field_name,
        threshold=threshold,
        verbose=verbose,
        cluster_algorithm='greedy_cover_set',
        sim_df=sim_df,
        filter_smaller=filter_smaller
    )
    train, test, valid = smallest_assignment(
        clusters, labels, size,
        valid_size, test_size
    )

    # Verbose output
    if verbose > 2:
        print(f'Proportion train: {(len(train) / size) * 100:.2f} %')
        print(f'Proportion test: {(len(test) / size) * 100:.2f} %')
        print(f'Proportion valid: {(len(valid) / size) * 100:.2f} %')

    # Warnings if the sizes of partitions are smaller than expected
    if len(test) < expected_test * 0.9 and verbose > 1:
        print(f'Warning: Proportion of test partition is smaller than expected: {(len(test) / size) * 100:.2f} %')
    if len(valid) < expected_valid * 0.9 and verbose > 1:
        print(f'Warning: Proportion of validation partition is smaller than expected: {(len(valid) / size) * 100:.2f} %')

    if valid_size > 0:
        return train, test, valid, clusters
    else:
        return train, test, clusters


def bitbirch(
    df: pd.DataFrame,
    sim_df: pd.DataFrame = None,
    field_name: str = None,
    label_name: str = None,
    test_size: float = 0.2,
    valid_size: float = 0.0,
    threshold: float = 0.3,
    verbose: int = 0,
    branching_factor: int = 50,
    n_bins: int = 10,
    radius: int = 2,
    bits: int = 1024,
    n_clusters: int = 20,
    **kwargs
):
    """Partition a dataset using the BitBirch clustering algorithm.

    Generates clusters based on molecular features or similarity using
    the BitBirch algorithm. Labels can be optionally discretized for
    balancing, and the dataset is partitioned into train/test/validation
    subsets using ``smallest_assignment``. Prints warnings if the partition
    sizes deviate from expectations.

    Reference: Pérez KL, Jung V, Chen L, Huddleston K, Miranda-Quintana RA.
    BitBIRCH: efficient clustering of large molecular libraries. Digital Discovery.
    2025;4(4):1042-51.

    :param df: DataFrame containing molecular entities.
    :type df: pd.DataFrame
    :param sim_df: Optional DataFrame containing pairwise similarity scores
        between entities, defaults to None.
    :type sim_df: pd.DataFrame, optional
    :param field_name: Optional column name used for clustering.
    :type field_name: str, optional
    :param label_name: Optional column name for labels used for balancing,
        defaults to None.
    :type label_name: str, optional
    :param test_size: Proportion of entities to allocate to the test subset,
        defaults to 0.2.
    :type test_size: float, optional
    :param valid_size: Proportion of entities to allocate to the validation
        subset, defaults to 0.0.
    :type valid_size: float, optional
    :param threshold: Similarity threshold used for clustering, defaults to 0.3.
    :type threshold: float, optional
    :param verbose: Verbosity level. Higher values print detailed partition
        proportions, defaults to 0.
    :type verbose: int, optional
    :param branching_factor: Branching factor for BitBirch clustering, defaults to 50.
    :type branching_factor: int, optional
    :param n_bins: Number of bins used when discretizing labels for balancing,
        defaults to 10.
    :type n_bins: int, optional
    :param radius: Neighborhood radius for BitBirch clustering, defaults to 2.
    :type radius: int, optional
    :param bits: Number of bits used in fingerprint representation, defaults to 1024.
    :type bits: int, optional
    :param n_clusters: Number of clusters to generate, defaults to 20.
    :type n_clusters: int, optional

    :return: If ``valid_size > 0`` returns train, test, valid subsets plus cluster assignments.
        Otherwise returns train, test subsets plus cluster assignments.
    :rtype: Union[
        Tuple[List[int], List[int], List[int], np.ndarray],
        Tuple[List[int], List[int], np.ndarray]
    ]
    """
    size = len(df)
    expected_test = test_size * size
    expected_valid = valid_size * size

    labels = df[label_name].to_numpy() if label_name else None
    labels = _discretizer(labels, n_bins=n_bins)

    # Generate cluster assignments
    clusters = generate_clusters(
        df,
        sim_df=sim_df,
        field_name=field_name,
        threshold=threshold,
        verbose=verbose,
        cluster_algorithm='bitbirch',
        radius=radius,
        branching_factor=branching_factor,
        bits=bits,
        n_clusters=n_clusters
    )
    clusters = np.array(clusters)
    train, test, valid = smallest_assignment(
        clusters, labels, size,
        valid_size, test_size
    )
    # Verbose output
    if verbose > 2:
        print(f'Proportion train: {(len(train) / size) * 100:.2f} %')
        print(f'Proportion test: {(len(test) / size) * 100:.2f} %')
        print(f'Proportion valid: {(len(valid) / size) * 100:.2f} %')

    # Warnings if the sizes of partitions are smaller than expected
    if len(test) < expected_test * 0.9 and verbose > 1:
        print(f'Warning: Proportion of test partition is smaller than expected: {(len(test) / size) * 100:.2f} %')
    if len(valid) < expected_valid * 0.9 and verbose > 1:
        print(f'Warning: Proportion of validation partition is smaller than expected: {(len(valid) / size) * 100:.2f} %')

    if valid_size > 0:
        return train, test, valid, clusters
    else:
        return train, test, clusters

from multiprocessing import cpu_count
from typing import Tuple, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from hestia.similarity import sim_df2mtx
from hestia.clustering import generate_clusters
from hestia.reduction import similarity_reduction
from hestia.utils import (_assign_partitions, _cluster_reassignment,
                          _neighbour_analysis, _balanced_labels,
                          limited_agglomerative_clustering, _discretizer)


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

    unique_parts, part_counts = np.unique(clusters, return_counts=True)
    sorted_parts = unique_parts[np.argsort(part_counts)]

    # Initialize empty lists for train, test, and valid sets
    test = []
    valid = []
    train = []

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
    # if sim_df is None:
    #     sim_df = calculate_similarity(
    #         df, df, data_type=data_type,
    #         similarity_metric=similarity_metric,
    #         field_name=field_name, threshold=threshold,
    #         threads=threads, verbose=verbose,
    #         save_alignment=False, filename=None, distance=distance,
    #         bits=bits, denominator=denominator, radius=radius,
    #         representation=representation, config=config
    #     )

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

import logging

from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from itertools import product

import numpy as np
import pandas as pd
import pickle as pk
import polars as pl

from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from hestia.partition import (
    ccpart, cdhit_part, sim_umap, perimeter_split, maximum_dissimilarity,
    butina
)
from hestia.utils.evaluation import evaluate
from hestia.utils.messages_cli import define_logger, welcome_autohestia
from tqdm import tqdm

AVAILABLE_ALGORITHMS = {
    'ccpart': ccpart,
    "cdhit": cdhit_part,
    "sim-umap": sim_umap,
    "perimeter_split": perimeter_split,
    "maximum_dissimilarity": maximum_dissimilarity,
    "butina": butina
}


class AutoHestia:
    """Automatic benchmarking and selection of dataset partitioning strategies.

    Sweeps all combinations of built-in (or custom) partitioning algorithms and
    pre-computed similarity metrics, evaluates each combination using a
    k-nearest neighbours probe across similarity thresholds from 0.1 to 1.0,
    and ranks them by monotonicity and mean performance. A guardrail filter then
    retains only experiments that meet minimum test-size and dynamic-range
    requirements, returning the top-ranked partitions ready for model training.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        field_name: str,
        x: np.ndarray,
        y: np.ndarray,
        sim_dfs: Dict[str, pl.DataFrame],
        verbose_level: str = 'debug'
    ):
        """Initialise AutoHestia.

        :param df: DataFrame containing the dataset entities.
        :param field_name: Column name in ``df`` that identifies each entity
            (e.g. sequence, SMILES, or PDB path).
        :param x: Pre-computed feature matrix of shape ``(n_samples, n_features)``
            used to train and evaluate the KNN probe.
        :param y: Label array of shape ``(n_samples,)``. Arrays with fewer than
            10 unique values are treated as classification; otherwise regression.
        :param sim_dfs: Mapping of metric name → pre-computed pairwise similarity
            Polars DataFrame (as returned by the ``hestia.similarity`` functions).
        :param verbose_level: Logging verbosity. One of ``'debug'``, ``'info'``,
            ``'warning'``, or ``'error'``. Defaults to ``'debug'``.
        """
        self.logger = define_logger('autohestia')

        if verbose_level.lower() == 'debug':
            self.logger.setLevel(logging.DEBUG)
        elif verbose_level.lower() == 'info':
            self.logger.setLevel(logging.INFO)
        elif verbose_level.lower() == 'warning':
            self.logger.setLevel(logging.WARNING)
        else:
            self.logger.setLevel(logging.ERROR)

        self.logger.info(welcome_autohestia())
        self.logger.info(
            f"\nDataset size: {len(df):,}"
        )
        self.df = df
        self.x = x
        self.y = y
        self.sim_dfs = sim_dfs
        self.fn = field_name
        self.task_type = 'c' if len(np.unique(y)) < 10 else 'r'
        self.metric = None

    def plot_good_curves(
        self,
        save_dir: str = 'tmp',
        overwrite: bool = False
    ):
        """Plot GOOD curves for all evaluated algorithm–metric combinations.

        Saves a ``good.png`` figure under ``<save_dir>/figures/`` and returns
        the Matplotlib figure object.

        :param save_dir: Root directory where the ``figures/`` sub-folder will
            be created. Defaults to ``'tmp'``.
        :param overwrite: If ``True``, an existing ``figures/`` directory is
            reused without raising an error. Defaults to ``False``.
        :raises RuntimeError: If called before :meth:`best_guardrailed_splits`.
        :return: Matplotlib Figure with one line per algorithm–metric combination.
        :rtype: matplotlib.figure.Figure
        """
        if self.metric is None:
            raise RuntimeError(
                "Before computing the plots, first is necessary to run the experiments."
            )
        import seaborn as sns
        import matplotlib.pyplot as plt

        save_dir = Path(save_dir) / "figures"
        save_dir.mkdir(exist_ok=overwrite)
        self.raw_experiments['combs'] = self.raw_experiments['part-alg'] + '+' + self.raw_experiments['sim-metric']
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(
            self.raw_experiments,
            x='th',
            y=self.metric,
            hue='combs'
        )
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        fig.tight_layout()
        fig.savefig(save_dir / 'good.png', bbox_inches='tight')
        return fig

    def best_guardrailed_splits(
        self,
        part_algs: Optional[List[str]] = None,
        custom_algs: Optional[Dict[str, Callable]] = None,
        top_k_parts: int = 3,
        top_l_sims: int = 3,
        min_test_size: float = 0.185,
        min_dynamic_range: float = 0.4,
        overwrite: bool = False,
        save_dir: str = 'tmp'
    ) -> dict:
        """Run the full guardrailed partitioning sweep and return the best splits.

        For every (partitioning algorithm, similarity metric) pair, partitions
        are generated at thresholds 0.1–1.0. Each partition is evaluated with a
        KNN probe (MCC for classification, Spearman CC for regression). Results
        are then filtered and ranked:

        1. **Dynamic-range filter** – combinations whose threshold range spans
           less than ``min_dynamic_range`` are discarded.
        2. **Top-k algorithms** – the ``top_k_parts`` algorithms with the
           highest worst-case mean performance are retained.
        3. **Top-l similarities** – for each retained algorithm, the
           ``top_l_sims`` similarity metrics with the best mean performance
           are kept.

        Intermediate results are written to ``<save_dir>/`` as CSV files and
        partition pickle files.

        :param part_algs: List of algorithm names to include. Must be keys of
            ``AVAILABLE_ALGORITHMS`` (``'ccpart'``, ``'cdhit'``,
            ``'sim-umap'``, ``'perimeter_split'``, ``'maximum_dissimilarity'``,
            ``'butina'``). Defaults to all available algorithms.
        :param custom_algs: Optional mapping of name → callable for
            user-supplied partitioning functions. The callable must accept the
            same keyword arguments as the built-in partition functions
            (``df``, ``sim_df``, ``field_name``, ``threshold``).
        :param top_k_parts: Number of top-performing partitioning algorithms to
            retain after the guardrail filter. Defaults to ``3``.
        :param top_l_sims: Number of similarity metrics to retain per algorithm.
            Defaults to ``3``.
        :param min_test_size: Minimum fraction of the dataset that must fall in
            the test split for a threshold to be included. Defaults to ``0.185``.
        :param min_dynamic_range: Minimum required span of valid thresholds for
            an algorithm–metric combination to pass the guardrail. Defaults to
            ``0.4``.
        :param overwrite: If ``True``, existing output directories are reused.
            Defaults to ``False``.
        :param save_dir: Root output directory. Defaults to ``'tmp'``.
        :raises ValueError: If ``part_algs`` contains an unrecognised algorithm
            name, or if no combinations survive the dynamic-range guardrail.
        :return: Dictionary with the following keys:

            - ``'raw-experiments'`` – :class:`pandas.DataFrame` of per-threshold
              KNN results for every combination.
            - ``'main-stats'`` – :class:`pandas.DataFrame` of aggregated
              statistics (monotonicity, dynamic range, mean performance) after
              the guardrail filter.
            - ``'after-guardrail'`` – subset of ``'main-stats'`` containing only
              the top-k × top-l combinations.
            - ``'top-combination'`` – ``(part_alg, sim_metric)`` tuple
              identifying the single best combination.
            - ``'best-parts'`` – dict mapping similarity threshold → ``{'train':
              np.ndarray, 'test': np.ndarray}`` for the top combination.
        :rtype: dict
        """
        save_dir = Path(save_dir)
        save_parts = save_dir / "parts"
        save_dir.mkdir(exist_ok=overwrite)
        save_parts.mkdir(exist_ok=overwrite)

        if part_algs is None:
            part_algs = list(AVAILABLE_ALGORITHMS.keys())
        else:
            for p in part_algs:
                if p not in AVAILABLE_ALGORITHMS:
                    raise ValueError(
                        f"Algorithm {p} not supported.",
                        f"Please use one of the following: {', '.join(AVAILABLE_ALGORITHMS)}",
                        "Otherwise declare it as a `custom_algs`."
                    )

        algs = deepcopy(AVAILABLE_ALGORITHMS)
        if custom_algs is not None:
            algs.update(custom_algs)

        if len(algs) < top_k_parts:
            self.logger.warning(
            (f"top_k_parts ({top_k_parts}) < number of algorithms ({len(algs)}.\n" +
                f"Defaulting to n algorithms: {len(algs)}")
            )
        if len(self.sim_dfs) < top_l_sims:
            self.logger.warning(
            (f"top_l_sims ({top_l_sims}) > number of sim metrics ({len(self.sim_dfs)}).\n" +
                f"Defaulting to n sim-metrics: {len(self.sim_dfs)}")
            )
        comb = list(product(part_algs, list(self.sim_dfs.keys())))
        pbar = tqdm(comb, total=len(comb))
        results = []
        for part_alg, sim_df in pbar:
            pbar.set_description(f"{part_alg} - {sim_df}")
            parts = {}
            for th in range(10, 110, 10):
                train, test, clusters = algs[part_alg](
                    df=self.df,
                    sim_df=self.sim_dfs[sim_df],
                    field_name=self.fn,
                    threshold=th/100
                )
                if len(test) < min_test_size * len(self.df):
                    continue
                parts[th/100] = {'train': train, 'test': test}
                knn = KNeighborsClassifier() if self.task_type == 'c' else KNeighborsRegressor()
                knn.fit(self.x[train], self.y[train])
                preds = knn.predict(self.x[test])
                result = evaluate(
                    preds, self.y[test],
                    pred_task='class' if self.task_type == 'c' else 'reg'
                )
                result['th'] = th/100
                result['part-alg'] = part_alg
                result['sim-metric'] = sim_df
                results.append(result)
            outpath = save_parts / f'{part_alg}-{sim_df}.pckl'
            pk.dump(parts, outpath.open('wb'))

        result_df = pd.DataFrame(results)
        m_results = []
        metric = 'mcc' if self.task_type == 'c' else 'spcc'
        self.metric = metric
        for (pa, sf), r_df in result_df.groupby(['part-alg', 'sim-metric']):
            result = {
                'pa': pa,
                'sf': sf,
                'monotonicity': spearmanr(r_df[metric], r_df['th']).statistic,
                'dynamic_range': r_df['th'].max() - r_df['th'].min(),
                'mean_perf': r_df[metric].mean()
            }
            m_results.append(result)
        m_df = pd.DataFrame(m_results)
        m_df = m_df[m_df['dynamic_range'] >= min_dynamic_range].reset_index(drop=True)
        if len(m_df) < 1:
            raise ValueError(
                "Dataset does not have enough samples after filtering for experiments with",
                f"dynamic range < {min_dynamic_range}. You may try with a smaller value."
            )
        top_k_pa = (
            m_df.groupby('pa')['mean_perf']
            .min()
            .nlargest(top_k_parts)
            .index
        )
        subset = m_df[m_df['pa'].isin(top_k_pa)]
        subset_top3_sf = (
            subset.sort_values('mean_perf', ascending=True)
            .groupby('pa')
            .head(top_l_sims)
        )
        subset_top3_sf.sort_values("monotonicity", inplace=True, ascending=False)
        subset_top3_sf.reset_index(inplace=True, drop=True)
        best_parts = pk.load((save_parts / f"{subset.iloc[0, 0]}-{subset.iloc[0, 1]}.pckl").open('rb'))
        n = {}
        for k, d in best_parts.items():
            n[k] = {sk: np.stack(sd) for sk, sd in d.items()}
        best_parts = n
        self.raw_experiments = result_df
        self.raw_experiments.to_csv(save_dir / "raw_experiments.csv", index=False)
        m_df.to_csv(save_dir / "main_stats.csv", index=False)
        output = {
            'raw-experiments': result_df,
            'main-stats': m_df,
            'after-guardrail': subset_top3_sf,
            'top-combination': (subset.iloc[0, 0], subset.iloc[0, 1]),
            'best-parts': best_parts
        }
        return output

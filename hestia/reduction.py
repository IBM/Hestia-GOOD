import numpy as np
import pandas as pd
import polars as pl

from hestia.clustering import generate_clusters


def similarity_reduction(
    df: pd.DataFrame,
    sim_df: pl.DataFrame,
    field_name: str,
    clustering_mode: str = "CDHIT",
    threshold: float = 0.9,
    verbose: int = 2,
) -> pl.DataFrame:
    if 'connected_components' == clustering_mode:
        mssg = 'Similarity reduction is not'
        mssg += ' implemented for connected components.'
        raise NotImplementedError(mssg)

    clusters = generate_clusters(df=df, field_name=field_name,
                                    sim_df=sim_df, verbose=verbose,
                                    threshold=threshold,
                                    cluster_algorithm=clustering_mode)
    representatives = np.unique(clusters)
    df = df[df.index.isin(representatives)]
    return df

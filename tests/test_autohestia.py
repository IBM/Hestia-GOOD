import os.path as osp

import numpy as np
import pandas as pd

from hestia.autohestia import AutoHestia


def test_autohestia():
    df = pd.read_csv(osp.join(
        osp.dirname(osp.realpath(__file__)), 'biogen_logS.csv')
    )
    df = df[~df['SMILES'].isna()].reset_index(drop=True)
    hestia = AutoHestia(
        df=df.iloc[:100],
        field_name='SMILES',
        label_name='logS',
        task_type='regression',
        data_type='molecule',
        representation='ecfp-4',
        verbose_level='debug',
        outdir=osp.join(osp.dirname(osp.realpath(__file__)), 'test_autohestia')
    )
    out = hestia.run()

    assert 'train' in out and 'test' in out
    assert isinstance(out, dict)
    assert isinstance(out['train'], np.ndarray)
    assert osp.exists(osp.join(hestia.outdir, 'parts-results.tsv'))
    assert osp.exists(osp.join(hestia.outdir, 'parts', 'ccpart-molformer.pckl'))
    assert osp.exists(osp.join(hestia.outdir, 'parts', 'ccpart-ecfp-4-t.pckl'))
    assert osp.exists(osp.join(hestia.outdir, 'parts', 'ccpart-mapc-4-j.pckl'))
    assert osp.exists(osp.join(hestia.outdir, 'parts', 'butina-molformer.pckl'))
    assert osp.exists(osp.join(hestia.outdir, 'parts', 'butina-ecfp-4-t.pckl'))
    assert osp.exists(osp.join(hestia.outdir, 'parts', 'butina-mapc-4-j.pckl'))

# def test_autohestia_good():
#     df = pd.read_csv(osp.join(
#         osp.dirname(osp.realpath(__file__)), 'biogen_logS.csv')
#     )
#     df = df[~df['SMILES'].isna()].reset_index(drop=True)
#     hestia = AutoHestia(
#         df=df.iloc[:100],
#         field_name='SMILES',
#         label_name='logS',
#         task_type='regression',
#         data_type='molecule',
#         representation='ecfp-4',
#         verbose_level='debug',
#         outdir=osp.join(osp.dirname(osp.realpath(__file__)), 'test_autohestia')
#     )
#     out = hestia.run()

#     assert 'train' in out and 'test' in out
#     assert isinstance(out, dict)
#     assert isinstance(out['train'], np.ndarray)

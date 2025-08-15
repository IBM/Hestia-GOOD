from typing import Callable

import numpy as np


def get_fp_function(
    fp: int,
    bits: int,
    radius: int,
    sim_function: str
) -> Callable:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import RDLogger
    from rdkit import rdBase

    def disable_rdkit_log():
        """Disable all rdkit logs."""
        for log_level in RDLogger._levels:
            rdBase.DisableLog(log_level)

    disable_rdkit_log()

    if fp == 'scaffold-ecfp':
        fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=bits
        )

        def _get_fp(smile: str):
            mol = Chem.MolFromSmiles(smile, sanitize=True)

            if mol is None:
                print(f"SMILES: `{smile}` could not be processed. Will be substituted by `C`")
                return _get_fp("C")
            mol = MurckoScaffold.GetScaffoldForMol(mol)

            if sim_function in ['dice', 'tanimoto', 'sokal', 'rogot-goldberg',
                                'cosine']:
                fp = fpgen.GetFingerprint(mol)
            else:
                fp = fpgen.GetFingerprintAsNumPy(mol).astype(np.int8)
            return fp

    elif fp == 'ecfp':
        fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=bits
        )

        def _get_fp(smile: str):
            mol = Chem.MolFromSmiles(smile, sanitize=True)

            if mol is None:
                print(f"SMILES: `{smile}` could not be processed. Will be substituted by `C`")
                return _get_fp("C")

            if sim_function in ['dice', 'tanimoto', 'sokal', 'rogot-goldberg',
                                'cosine']:
                fp = fpgen.GetFingerprint(mol)
            else:
                fp = fpgen.GetFingerprintAsNumPy(mol).astype(np.int8)
            return fp

    elif fp == 'scaffold-maccs':
        def _get_fp(smile: str):
            mol = Chem.MolFromSmiles(smile)

            if mol is None:
                print(f"SMILES: `{smile}` could not be processed. Will be substituted by `C`")
                return _get_fp("C")
            mol = MurckoScaffold.GetScaffoldForMol(mol)

            fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            if sim_function in ['dice', 'tanimoto', 'sokal', 'rogot-goldberg',
                                'cosine']:
                return fp
            else:
                return np.array(fp)

    elif fp == 'maccs':
        def _get_fp(smile: str):
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                print(f"SMILES: `{smile}` could not be processed. Will be substituted by `C`")
                return _get_fp("C")

            fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            if sim_function in ['dice', 'tanimoto', 'sokal', 'rogot-goldberg',
                                'cosine']:
                return fp
            else:
                return np.array(fp)

    elif fp == 'scaffold-mapc':
        try:
            from mapchiral.mapchiral import encode
        except ModuleNotFoundError:
            raise ImportError('This fingerprint requires mapchiral to be installed: `pip install mapchiral`')

        def _get_fp(smile: str):
            mol = Chem.MolFromSmiles(smile, sanitize=True)
            if mol is None:
                print(f"SMILES: `{smile}` could not be processed. Will be substituted by `C`")
                return _get_fp("C")
            try:
                mol2 = MurckoScaffold.GetScaffoldForMol(mol)
                Chem.SanitizeMol(mol2)
                fp = encode(mol2, max_radius=radius,
                            n_permutations=bits, mapping=False)
            except RuntimeError as e:
                print(f"Warning: {e}")
                fp = encode(mol, max_radius=radius,
                            n_permutations=bits, mapping=False)
            return fp

        if sim_function != 'jaccard':
            raise ValueError('MAPc can only be used with `jaccard`.')

    elif fp == 'mapc':
        try:
            from mapchiral.mapchiral import encode
        except ModuleNotFoundError:
            raise ImportError('This fingerprint requires mapchiral to be installed: `pip install mapchiral`')

        def _get_fp(smile: str):
            mol = Chem.MolFromSmiles(smile, sanitize=True)
            if mol is None:
                print(f"SMILES: `{smile}` could not be processed. Will be substituted by `C`")
                return _get_fp("C")

            fp = encode(mol, max_radius=radius,
                        n_permutations=bits, mapping=False)
            return fp

        if sim_function != 'jaccard':
            raise ValueError('MAPc can only be used with `jaccard`.')

    elif fp == 'lipinski':
        import rdkit.Chem.Lipinski as Lip

        def _get_fp(smiles: str):
            fp = []
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
            if mol is None:
                print(f"SMILES: `{smiles}` could not be processed. Will be substituted by `C`")
                return _get_fp("C")
            fp.append(Lip.NumHAcceptors(mol))
            fp.append(Lip.NumHDonors(mol))
            fp.append(Lip.NumHeteroatoms(mol))
            fp.append(Lip.NumRotatableBonds(mol))
            fp.append(Lip.NumSaturatedCarbocycles(mol))
            fp.append(Lip.RingCount(mol))

            return np.array(fp)

        if sim_function != 'canberra':
            raise ValueError("Lipinski can only be used with sim_function='canberra'")
    return _get_fp

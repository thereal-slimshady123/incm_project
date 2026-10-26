"""
Optimization and Model Fitting module: QMP loss, Nelder-Mead fitting engine, and CRN harness.
"""

from .qmp import QMPDataset, compute_qmp_nll, compute_bic, compute_aic
from .fit import fit_ddm, DDMFitter

__all__ = ["QMPDataset", "compute_qmp_nll", "compute_bic", "compute_aic", "fit_ddm", "DDMFitter"]

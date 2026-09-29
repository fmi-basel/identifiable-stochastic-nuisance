

from implicit_nuisance.methods.base import BaseMethod
from implicit_nuisance.methods.infoLDM import InfoLDM
from implicit_nuisance.methods.infoLDM_action import InfoLDMAction
from implicit_nuisance.methods.next_observation import NextObservation

METHODS = {
    # base classes
    "base": BaseMethod,
    # methods
    "infoLDM": InfoLDM,
    "infoLDMAction": InfoLDMAction,
    "nextObservation": NextObservation,
    # Legacy checkpoints in the adapted repository saved this obsolete name.
    "qalmanSSL": InfoLDM,
}
__all__ = [
    "InfoLDM",
    "InfoLDMAction",
    "NextObservation",
]

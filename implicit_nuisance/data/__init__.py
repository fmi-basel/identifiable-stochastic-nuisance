


from implicit_nuisance.data.causal3dident import Causal3DIdent
from implicit_nuisance.data.dm_control import DMControlReacherNuisance
from implicit_nuisance.data.dm_control_hopper_rollout import (
    DMControlHopperNuisance,
)
from implicit_nuisance.data.offline_dm_control import OfflineDMControlReacherNuisance
from implicit_nuisance.data.offline_dm_control_hopper_rollout import (
    OfflineDMControlHopperNuisance,
)
from implicit_nuisance.data.dot_motion import DotMotion
from implicit_nuisance.data.random_step import RandomStep
from implicit_nuisance.data.random_step_dirichlet_nuisance import (
    RandomStepDirichletNuisance,
)
from implicit_nuisance.data.random_step_nuisance import RandomStepNuisance

DATASETS = {
    "causal3dident": Causal3DIdent,
    "dm_control_reacher_nuisance": DMControlReacherNuisance,
    "dm_control_hopper_nuisance": DMControlHopperNuisance,
    "offline_dm_control_reacher_nuisance": OfflineDMControlReacherNuisance,
    "offline_dm_control_hopper_nuisance": OfflineDMControlHopperNuisance,
    "dot_motion": DotMotion,
    "randstep": RandomStep,
    "randstep_dirichlet_nuisance": RandomStepDirichletNuisance,
    "randstep_nuisance": RandomStepNuisance,
}


__all__ = [
    "causal3dident",
    "dm_control_reacher_nuisance",
    "dm_control_hopper_nuisance",
    "offline_dm_control_reacher_nuisance",
    "offline_dm_control_hopper_nuisance",
    "dot_motion",
    "randstep",
    "randstep_dirichlet_nuisance",
    "randstep_nuisance",
]

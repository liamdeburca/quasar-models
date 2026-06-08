from astropy.modeling import CompoundModel
from quasar_models.tying.base_parameter_tie import BaseParameterTie

class IdenticalTie(BaseParameterTie):
    def __call__(self, model: CompoundModel) -> float:
        param = model[self.target_name]
        return getattr(param, self.target_parameter).value
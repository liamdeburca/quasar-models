from astropy.modeling import Parameter

from ..linear_tie import LinearTie


def param_is_fixed(param: Parameter) -> bool:
    return param.fixed


def param_is_tied(param: Parameter) -> bool:
    tied = param.tied
    if tied in (False, None):
        return False
    elif isinstance(tied, LinearTie):
        return True
    else:
        msg = f"Parameter '{param.name}' has invalid 'tied' attribute: {tied}."
        raise TypeError(msg)


def param_is_free(param: Parameter) -> bool:
    return not param_is_fixed(param) and not param_is_tied(param)

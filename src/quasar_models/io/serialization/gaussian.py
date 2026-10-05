from quasar_utils.setup import Info

from quasar_models.line import GaussianModel


def serialize_gaussian(model: GaussianModel, info: Info):
    return model.serialize(info)


def deserialize_gaussian(data: dict, name: str, info: Info) -> GaussianModel:
    return GaussianModel.deserialize(data, name, info)

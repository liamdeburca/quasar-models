from quasar_utils.setup import Info

from quasar_models.continuum import PowerLawModel


def serialize_powerlaw(model: PowerLawModel, info: Info):
    return model.serialize(info)


def deserialize_powerlaw(data: dict, name: str, info: Info) -> PowerLawModel:
    return PowerLawModel.deserialize(data, name, info)

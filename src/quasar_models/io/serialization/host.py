from quasar_utils.setup import Info

from quasar_models.host import HostGalaxyModel


def serialize_host(model: HostGalaxyModel, info: Info):
    return model.serialize(info)


def deserialize_host(data: dict, name: str, info: Info) -> HostGalaxyModel:
    return HostGalaxyModel.deserialize(data, name, info)

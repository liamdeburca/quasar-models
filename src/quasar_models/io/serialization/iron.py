from quasar_utils.setup import Info

from quasar_models.iron import IronModel


def serialize_iron(model: IronModel, info: Info):
    return model.serialize(info)


def deserialize_iron(data: dict, name: str, info: Info) -> IronModel:
    return IronModel.deserialize(data, name, info)

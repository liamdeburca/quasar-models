from quasar_utils.setup import Info

from quasar_models.balmer import BalmerModel


def serialize_balmer(model: BalmerModel, info: Info):
    return model.serialize(info)

def deserialize_balmer(data: dict, name: str, info: Info) -> BalmerModel:
    return BalmerModel.deserialize(data, name, info)

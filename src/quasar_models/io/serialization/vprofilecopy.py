from quasar_utils.setup import Info

from quasar_models.line.vprofilecopies import _VProfileCopy


def serialize_vprofilecopy(model: _VProfileCopy, info: Info):
    return model.serialize(info)


def deserialize_vprofilecopy(data: dict, name: str, info: Info) -> _VProfileCopy:
    return _VProfileCopy.deserialize(data, name, info)

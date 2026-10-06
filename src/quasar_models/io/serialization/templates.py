from quasar_utils.setup import Info

from quasar_models.balmer.continuum import BalmerContinuumTemplate
from quasar_models.balmer.series import BalmerSeriesTemplate
from quasar_models.host import HostGalaxyTemplate
from quasar_models.iron import IronTemplate


def serialize_iron_template(template: IronTemplate, info: Info):
    return template.serialize(info)


def deserialize_iron_template(data: dict, info: Info) -> IronTemplate:
    return IronTemplate.deserialize(data, info)


def serialize_host_template(template: HostGalaxyTemplate, info: Info):
    return template.serialize(info)


def deserialize_host_template(data: dict, info: Info) -> HostGalaxyTemplate:
    return HostGalaxyTemplate.deserialize(data, info)


def serialize_balmer_continuum_template(template: BalmerContinuumTemplate, info: Info):
    return template.serialize(info)


def deserialize_balmer_continuum_template(data: dict, info: Info) -> BalmerContinuumTemplate:
    return BalmerContinuumTemplate.deserialize(data, info)


def serialize_balmer_series_template(template: BalmerSeriesTemplate, info: Info):
    return template.serialize(info)


def deserialize_balmer_series_template(data: dict, info: Info) -> BalmerSeriesTemplate:
    return BalmerSeriesTemplate.deserialize(data, info)

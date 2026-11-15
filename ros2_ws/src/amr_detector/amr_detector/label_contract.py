"""Names and confidence thresholds follow the loaded model's ontology."""
from collections.abc import Mapping


THRESHOLDS_BY_NAME = {
    'person': 0.30,
    'rider': 0.25,
    'fork_truck': 0.25,
    'cart_pallet_jack': 0.35,
    'rack_shelf': 0.30,
    'misc_dynamic_feature': 0.35,
}


def checkpoint_classes(names):
    if isinstance(names, Mapping):
        converted = {int(key): value for key, value in names.items()}
        if len(converted) != len(names):
            raise ValueError('Duplicate model class IDs')
    elif isinstance(names, (list, tuple)):
        converted = dict(enumerate(names))
    else:
        raise ValueError('Model class metadata must be a mapping or list')
    if not converted or set(converted) != set(range(len(converted))):
        raise ValueError('Model class IDs must be contiguous from zero')
    if any(not isinstance(value, str) or not value.strip() for value in converted.values()):
        raise ValueError('Model class names must be nonempty strings')
    return converted


def class_thresholds(names):
    return {key: THRESHOLDS_BY_NAME.get(value, 0.25) for key, value in names.items()}

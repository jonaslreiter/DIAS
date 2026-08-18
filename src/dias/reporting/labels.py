"""Display names for actions, relations and objects."""


def resolve_label_names(
    datamodule, num_actions: int, num_relations: int, num_objects: int
):
    """Resolve action / relation / object display names from a datamodule with numeric fallbacks."""

    def _normalise(names, target_len, prefix):
        lst = [] if names is None else list(names)
        if len(lst) < target_len:
            lst.extend(f"{prefix} {i}" for i in range(len(lst), target_len))
        return lst[:target_len]

    action_names = _normalise(
        getattr(datamodule, "action_names", None), num_actions, "Action"
    )
    relation_names = _normalise(
        getattr(datamodule, "relation_names", None), num_relations, "Rel"
    )

    obj_dict = getattr(datamodule, "object_names", None)
    if obj_dict is not None:
        object_names = [obj_dict.get(str(i), f"Node {i}") for i in range(num_objects)]
    else:
        object_names = [f"Node {n}" for n in range(num_objects)]

    return action_names, relation_names, object_names

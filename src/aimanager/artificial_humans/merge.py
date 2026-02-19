def deepmerge(base, new):
    if isinstance(base, dict) and isinstance(new, dict):
        return {**base, **new}
    elif isinstance(base, list) and isinstance(new, list):
        if len(base) == len(new):
            return [deepmerge(b, n) for b, n in zip(base, new)]
        else:
            ValueError("Length of list in deepmerge do not match.")
    else:
        return new


def deepermerge(base, new):
    if isinstance(base, dict) and isinstance(new, dict):
        merged = {k: deepermerge(b, new[k]) if k in new else b for k, b in base.items()}
        return {**new, **merged}
    elif isinstance(base, list) and isinstance(new, list):
        if len(base) == len(new):
            return [deepmerge(b, n) for b, n in zip(base, new)]
        else:
            ValueError("Length of list in deepmerge do not match.")
    else:
        return new

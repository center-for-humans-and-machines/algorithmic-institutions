def assoc_in(obj, keys, value):
    if len(keys) == 0:
        return value
    else:
        key = keys[0]
        if isinstance(obj, list):
            if key == "x":
                assert len(keys) == 1, 'Found appending ".x." in the middle of key.'
                if value is None:
                    return obj
                else:
                    return obj + [assoc_in(None, keys[1:], value)]
            else:
                idx = int(key)
                return (
                    obj[:idx] + [assoc_in(obj[idx], keys[1:], value)] + obj[idx + 1 :]
                )
        elif isinstance(obj, dict):
            return {**obj, key: assoc_in(obj.get(key, {}), keys[1:], value)}
        else:
            raise ValueError(f"Expected dict or list, got {obj}")


# def get_in(keys, obj):
#     if obj == None:
#         return None
#     if len(keys) == 0:
#         return obj
#     else:
#         key = keys[0]
#         if isinstance(obj, list):
#             if key == 'x':
#                 assert len(keys) == 1
#                 return None
#             else:
#                 idx = int(key)
#                 return get_in(keys[1:], obj[idx])
#         elif isinstance(obj, dict):
#             return get_in(keys[1:], obj.get(key))
#         else:
#             return None


def parse_simple(jobs, grid_dim):
    _jobs = []
    dim_length = get_dim_length(grid_dim.values())
    for job in jobs:
        for v_id in range(dim_length):
            _job = job
            for keys, values in grid_dim.items():
                split_keys = keys.split(".")
                value = values[v_id]
                _job = assoc_in(_job, split_keys, value)
            _jobs.append(_job)
    return _jobs


def get_dim_length(ll):
    lens = [len(length) for length in ll]
    assert min(lens) == max(lens), "Different lengths in same dimension."
    return min(lens)


def parse_list(jobs, grid_dim):
    _jobs = []
    for job in jobs:
        for grid_val in grid_dim:
            _job = job
            for keys, value in grid_val.items():
                k_list = keys.split(".")
                # old_v = get_in(k_list, job)
                # print(keys, old_v)
                # new_v = deepmerge(old_v, value)
                _job = assoc_in(_job, k_list, value)
            _jobs.append(_job)
    return _jobs


def parse_dim(jobs, grid_dim):
    if isinstance(grid_dim, dict):
        return parse_simple(jobs, grid_dim)
    elif isinstance(grid_dim, list):
        return parse_list(jobs, grid_dim)
    else:
        raise ValueError("Unkown grid type.")


def parse_grid(grid, base_job):
    if base_job.get("params_only"):
        jobs = [base_job["params"]]
    else:
        dummy = {"labels": {}, "params": {}}
        jobs = [{**dummy, **base_job}]
    for grid_dim in grid:
        jobs = parse_dim(jobs, grid_dim)
    if base_job.get("params_only"):
        jobs = [{**base_job, "params": p} for p in jobs]
    return jobs

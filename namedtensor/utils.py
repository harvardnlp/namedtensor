def make_tuple(names):
    if names is None:
        return ()
    if isinstance(names, tuple):
        return names
    if isinstance(names, list):
        return tuple(names)
    else:
        return (names,)

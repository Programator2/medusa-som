def find_transformation(original, new):
    """Find transformation from `original` path to `new` path. There are three
    possible transformations:

    1) '..' --- go up a folder

    2) 'dentry' --- go down to a `dentry` folder or file

    3) '' --- do nothing (only when `original == new`)

    """
    index = 0
    len_original = len(original)
    len_new = len(new)
    while original[index] == new[index]:
        index += 1
        if index >= len_original and index < len_new:
            # original is a prefix and it needs to go deeper
            index += 1  # go over forward slash
            try:
                dentry_end = new.index('/', index)
            except ValueError:
                return new[index:]
            return new[index:dentry_end]
        elif index >= len_new and index < len_original:
            # new is a prefix, orginal needs to go higher
            return '..'
        elif index >= len_original and index >= len_new:
            # paths are equal
            return ''
    return '..'

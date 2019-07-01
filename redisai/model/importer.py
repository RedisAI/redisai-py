
def load_model(path: str):
    """
    Return the binary data if saved with `as_native` otherwise return the dict
    that contains binary graph/model on `graph` key (Not implemented yet).
    :param path: File path from where the native model or the rai models are saved
    """
    with open(path, 'rb') as f:
        return f.read()


def load_script(path: str):
    """
    Read the script file as binary and return the content
    """
    with open(path, 'rb') as f:
        return f.read()

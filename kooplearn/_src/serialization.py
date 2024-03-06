import pickle
from pathlib import Path


def pickle_save(obj, filename, protocol=5):
    # Check if 'filename' is a path-like or a file-like object
    if hasattr(filename, "write"):
        # If it is a file-like object, we dump the object to it
        pickle.dump(obj, filename, protocol=protocol)
    else:
        # If it is a path-like object, we dump the object to a file, creating the folder structure, if it does not exist
        filename = Path(filename)
        filename.parent.mkdir(exist_ok=True, parents=True)
        with open(filename, "+wb") as outfile:
            pickle.dump(obj, outfile, protocol=protocol)


def pickle_load(base_cls, filename):
    # Check if 'filename' is a path-like or a file-like object
    if hasattr(filename, "read"):
        # If it is a file-like object, we load the object from it
        restored_obj = pickle.load(filename)
    else:
        # If it is a path-like object, we load the object from a file
        with open(filename, "+rb") as infile:
            restored_obj = pickle.load(infile)
    assert isinstance(restored_obj, base_cls)
    return restored_obj

import pickle


def save_to_pickle(obj, fpath):
    with open(fpath, 'wb') as file:
        # A new file will be created
        pickle.dump(obj, file)
    return None


def load_from_pickle(fpath):
    # Open the file in binary mode
    with open(fpath, 'rb') as file:
        # Call load method to deserialze
        obj = pickle.load(file)
    return obj

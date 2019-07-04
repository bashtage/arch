def dataset_loader(dataset):
    """Load a dataset using the new syntax is possible"""
    try:
        return dataset.load(as_pandas=True).data
    except TypeError:
        return dataset.load().data

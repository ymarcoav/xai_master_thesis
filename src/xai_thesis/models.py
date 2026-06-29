def build_model(model_name: str, random_state: int = 42):
    """Create a model by name."""
    # Add model registry entries as experiments are defined.
    raise NotImplementedError(f"Unknown or unimplemented model: {model_name}")


def train_model(model, x_train, y_train):
    """Fit a model and return it."""
    return model.fit(x_train, y_train)

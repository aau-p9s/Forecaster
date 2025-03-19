def inspect_model(model):
    """
    Inspects a trained model and prints all its attributes and their values.

    Parameters:
        model (object): The trained model instance.
    """
    try:
        # Retrieve all attributes
        attributes = vars(model)  # or model.__dict__

        # Print attributes and their values
        print("Model Attributes:")
        for attr_name, attr_value in attributes.items():
            print(f"{attr_name}: {attr_value}")
    except Exception as e:
        print(f"Error inspecting model: {e}")

LOCAL_SOURCES_FILEPATH = "./sources.txt"


def update_existing_sources(existing_sources):
    """Updates the list of existing sources to reflect the local sources file."""
    existing_sources.clear()

    with open(LOCAL_SOURCES_FILEPATH, 'r') as file:
        for line in file:
            existing_sources.append(line.strip())


def clear_local_sources():
    """Clears the local sources file."""
    with open(LOCAL_SOURCES_FILEPATH, 'w') as file:
        pass

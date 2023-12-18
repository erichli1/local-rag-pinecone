from constants import LOCAL_SOURCES_FILEPATH


def update_existing_sources(existing_sources):
    existing_sources.clear()

    with open(LOCAL_SOURCES_FILEPATH, 'r') as file:
        for line in file:
            existing_sources.append(line.strip())

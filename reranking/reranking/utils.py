import os


LOCATIONS = ["HYDRO", "SUCCULENT"]
SESSIONS = ["hl_map", "hl_query", "ios_map", "ios_query", "spot_map", "spot_query"]


def get_location_and_session(path: os.PathLike) -> tuple:
    dirs = path.split("/") #Not os.path. FUTURE FIXME

    location = ""
    for possible_loc in LOCATIONS:
        if possible_loc in dirs:
            location = possible_loc
            break

    session = ""
    for possible_session in SESSIONS:
        if possible_session in dirs:
            session = possible_session
            break

    return location, session

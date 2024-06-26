import json
import os


def save_to_json(dictionary, file_name):
    # Create directory if not present
    directory = os.path.dirname(file_name)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_name, "w") as f:
        json.dump(dictionary, f)


def load_from_json(file_name) -> dict:
    with open(file_name, "r") as f:
        return json.load(f)


def load_list(filename):
    with open(filename) as file:
        return [line.rstrip() for line in file]


def update_solutions_json():
    a = load_list("solutions/countries.txt")
    b = load_list("solutions/world_capitals.txt")
    c = load_list("solutions/states.txt")
    d = load_list("solutions/state_capitals.txt")
    e = load_list("solutions/presidents.txt")

    solutions = {
        "countries": a,
        "world_capitals": b,
        "states": c,
        "state_capitals": d,
        "presidents": e,
    }
    for key in solutions:
        solutions[key] = [s.lower() for s in solutions[key]]
    save_to_json(solutions, "solutions.json")


if __name__ == "__main__":
    update_solutions_json()

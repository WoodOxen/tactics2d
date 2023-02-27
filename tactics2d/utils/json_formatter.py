import json


def json_formatter(file):
    with open(file, "r") as f:
        data = json.load(f)
        f.close()

    with open(file, "w") as f:
        string = json.dumps(data, indent=2)
        f.write(string)
        f.close()


if __name__ == "__main__":
    json_formatter(
        "/home/rowena/Documents/PublicRepos/tactics2d/tactics2d/data/trajectory_sample/DLP/DJI_0012_scene.json"
    )

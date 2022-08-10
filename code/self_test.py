import json

import infer


if __name__ == "__main__":
    input_shapshot_path = "./example/input_snapshot.json"
    input_snapshot = json.load(open(input_shapshot_path))

    input_image = input_snapshot["yolo5_output"][0]["head"]

    model = infer.load_pretrained_model("./download/head_swin_bnneck")

    embedding = infer.get_embedding_for_json(model, input_image)

    print(embedding)
    print("Self test success")
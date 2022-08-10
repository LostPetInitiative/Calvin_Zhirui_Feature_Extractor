import json

import infer


if __name__ == "__main__":
    input_shapshot_path = "./example/input_snapshot.json"
    input_snapshot = json.load(open(input_shapshot_path))

    input_image = input_snapshot["yolo5_output"][0]["head"]

    device = 'cpu'

    model,config = infer.load_pretrained_model("./download/head_swin_bnneck", device)
    #print("Config")
    #print(config)
    preproc_transform = infer.get_infer_transform(config.image_size)
    embedding = infer.get_embedding_for_json(model, preproc_transform, input_image)

    print(embedding)
    assert embedding.shape[0] == 1
    assert embedding.shape[1] == 1024
    
    print("Self test success")
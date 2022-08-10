
from socket import J1939_NLA_BYTES_ACKED
import unittest
import json

import infer

device = 'cpu'

model,config = infer.load_pretrained_model("./head_swin_bnneck", device)
#print("Config")
#print(config)
preproc_transform = infer.get_infer_transform(config.image_size)

class EmbeddingsTest(unittest.TestCase):
    def test_embedding_produced(self):
        input_shapshot_path = "./example/input_snapshot.json"
        input_snapshot = json.load(open(input_shapshot_path))

        input_image = input_snapshot["yolo5_output"][0]["head"]

        embedding = infer.get_embedding_for_json(model, preproc_transform, input_image)

        print(embedding)
        assert embedding.shape[0] == 1
        assert embedding.shape[1] == 1024

    def test_process_job(self):
        input_shapshot_path = "./example/input_snapshot.json"
        job = json.load(open(input_shapshot_path))

        expected_output_path = "./example/expected_output_snapshot.json"
        expected_output = json.load(open(expected_output_path))

        output_job = infer.process_job(model, preproc_transform, job)

        exp_json = json.dumps(expected_output)
        actual_json = json.dumps(output_job)
        assert exp_json == actual_json, f"Output job does not match expected output {exp_json} {actual_json}"
        assert "yolo5_output" not in output_job
    
        

if __name__ == "__main__":
    unittest.main()
    print("Self test success")
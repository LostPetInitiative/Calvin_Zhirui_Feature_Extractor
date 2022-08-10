import os
import infer

import kafkajobs
from infer import process_job

kafkaUrl = os.environ['KAFKA_URL']
inputQueueName = os.environ['INPUT_QUEUE']
outputQueueName = os.environ['OUTPUT_QUEUE']

appName = "zhiru-calvin-head-swin-bnneck-feature-extractor"

worker = kafkajobs.jobqueue.JobQueueWorker(appName, kafkaBootstrapUrl=kafkaUrl, topicName=inputQueueName, appName=appName)
resultQueue = kafkajobs.jobqueue.JobQueueProducer(kafkaUrl, outputQueueName, appName)


device = 'cpu'
model,config = infer.load_pretrained_model("./head_swin_bnneck", device)
preproc_transform = infer.get_infer_transform(config.image_size)
print("model loaded")

def work():
    print("Service started. Pooling for a job")
    while True:        
        job = worker.GetNextJob(5000)
        uid = job["uid"]
        
        print("Got job {0}".format(uid))

        out_job = process_job(model,preproc_transform, job)
        
        resultQueue.Enqueue(uid, out_job)
        worker.Commit()
        print("{0}: Job processed successfully, results are submited to kafka".format(uid))

work()
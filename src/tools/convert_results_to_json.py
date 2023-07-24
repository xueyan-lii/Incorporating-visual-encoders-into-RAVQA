#convert results downloaded from wandb to json
#used for creating faiss index, the content isn't used for faiss, but inserted into the faiss process
#take all distinct predictions and corresponding scores
# calculate Recall where any prediction appear in any ground truth answers
# calculate hit rate the same as prophet
#k is only changed for evaluation purposes, also keep to max when generating files for faiss
import csv
import json
 
def csv_to_json(csv_file_path, json_file_path, K=100):
    data_dict = {}
    length = []
    with open(csv_file_path) as csv_file_handler:
        csv_reader = csv.DictReader(csv_file_handler)
        positive_counter, scores_counter = 0, 0
        total_counter = 0
        for rows in csv_reader:
            total_counter += 1
            row_dict={}
            key = rows['question_id']
            row_dict['image_key']=rows['image_key']
            row_dict['question']=rows['question']
            if rows['caption'][-1] != ".":
                row_dict['oscar_caption']=rows['caption'] + '.'
            else:
                row_dict['oscar_caption']=rows['caption']
            if blip2_captions[rows['image_key']] != ".":
                row_dict['blip2_caption']=blip2_captions[rows['image_key']] + "."
            else:
                row_dict['blip2_caption']=blip2_captions[rows['image_key']]
            row_dict['answers']=rows['answers'].split(',')
            row_dict['gold_answer']=rows['gold_answer']
            row_dict['prediction']=rows['prediction']
            predictions = rows['doc_predictions'].split(",")
            scores = rows['doc_scores'].split(",")

            predictions_indexes = [predictions.index(x) for x in set(predictions)]
            predictions_indexes.sort()
            #print(predictions_indexes)
            row_dict['doc_predictions']=[predictions[i] for i in predictions_indexes]
            row_dict['doc_scores'] = [scores[i] for i in predictions_indexes]
            if len(row_dict['doc_predictions']) > K:
                row_dict['doc_predictions'] = row_dict['doc_predictions'][:K]
                row_dict['doc_scores'] = row_dict['doc_scores'][:K]

            for i in row_dict['doc_predictions']:
                if i in row_dict['answers']:
                    positive_counter+=1
                    break
            scores_list=[]
            for i in row_dict['doc_predictions']:
                if i in row_dict['answers']:
                    no=row_dict['answers'].count(i)
                    soft_score=no/3.0
                    if soft_score>1:
                        soft_score=1
                    scores_list.append(soft_score)
            #print(scores)
            if scores_list != []:
                scores_counter+=max(scores_list)
                
            #print(rows['doc_predictions'], rows['doc_scores'])
            length.append(len(row_dict['doc_predictions']))
            data_dict[key] = row_dict

    for i in range(max(length)+1):
        print(i,"occured",length.count(i),'times')

    print('recall',positive_counter/float(total_counter),'hit rate',scores_counter/total_counter)
    with open(json_file_path, 'w') as json_file_handler:
        #Step 4
        json_file_handler.write(json.dumps(data_dict, indent = 4))
 

csv_file_path = "/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/testing_results/RAVQA-InstructBLIP-test-K50.csv"
json_file_path = "/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/testing_results/RAVQA-InstructBLIP-test-K50.json"
blip2_caption_path = "/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/pre-extracted_features/captions/coco_blip2_captions_val2014.json"

with open(blip2_caption_path) as json_file:
    blip2_captions = json.load(json_file)
#print(blip2_captions)
csv_to_json(csv_file_path, json_file_path, K=50)

csv_file_path = "/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/testing_results/RAVQA-InstructBLIP-train-K50.csv"
json_file_path = "/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/testing_results/RAVQA-InstructBLIP-train-K50.json"
blip2_caption_path = "/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/pre-extracted_features/captions/coco_blip2_captions_train2014.json"
with open(blip2_caption_path) as json_file:
    blip2_captions = json.load(json_file)
csv_to_json(csv_file_path, json_file_path, K=50)
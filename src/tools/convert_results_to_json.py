#convert results downloaded from wandb to json
#used for creating faiss index, the content isn't used for faiss, but inserted into the faiss process
#take all distinct predictions and corresponding scores
# calculate Recall where any prediction appear in any ground truth answers
# calculate hit rate the same as prophet
#k is only changed for evaluation purposes, also keep to max when generating files for faiss
import csv
import json
import random

#deal with the issue in reduced training set where some numbers are like 1,300
def csv_to_json(csv_file_path, json_file_path, K=100):
    data_dict = {}
    length = []
    exceed_counter=0
    
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

            if blip2_captions[rows['image_key']][-1] != ".":
                row_dict['blip2_caption']=blip2_captions[rows['image_key']] + "."
            else:
                row_dict['blip2_caption']=blip2_captions[rows['image_key']]

            if promptcap_captions[rows['question_id']][-1] != ".":
                row_dict['promptcap_caption']=promptcap_captions[rows['question_id']] + "."
            else:
                row_dict['promptcap_caption']=promptcap_captions[rows['question_id']]
            row_dict['answers']=rows['answers'].split(',')
            row_dict['gold_answer']=rows['gold_answer']
            row_dict['prediction']=rows['prediction']
            predictions = rows['doc_predictions'].split(",")
            #predictions=predictions[:5]#to limit total number of candidates
            scores = rows['doc_scores'].split(",")
            #cases where a number like 12,000 is given, deal with it case by case rather than write a function
            
            if len(predictions) != len(scores):
                #print(key)
                #print(len(predictions),len(scores))
                #change 1 by 1 in original csv until no more errors
                #print(predictions)
                corrected_predictions=[]
                i=0
                while i < len(predictions)-1:
                    if predictions[i][-1].isnumeric() and predictions[i+1][:3]=='000':
                        corrected_predictions.append(predictions[i]+predictions[i+1])
                        i+=2
                    else:
                        corrected_predictions.append(predictions[i])
                        i+=1
                if i==len(predictions)-1:
                    corrected_predictions.append(predictions[i])
                #print(corrected_predictions)
                if len(corrected_predictions)==50:
                    predictions=corrected_predictions
                else:#if still have issue then idk what is wrong either, as in i dont know which number if incorrectly split
                    print(key)
                    print(len(corrected_predictions), corrected_predictions)
                    predictions=corrected_predictions[:50]
                    exceed_counter+=1
                
            predictions_indexes = [predictions.index(x) for x in set(predictions)]
            predictions_indexes.sort()
            
            row_dict['doc_predictions']=[predictions[i] for i in predictions_indexes]
            row_dict['doc_scores'] = [scores[i] for i in predictions_indexes]
            if len(row_dict['doc_predictions']) > K:
                row_dict['doc_predictions'] = row_dict['doc_predictions'][:K]
                row_dict['doc_scores'] = row_dict['doc_scores'][:K]

            #randomly remove answers to intentionally worsen training ACRecall and hit rate, only for TRAIN!
            #only for those with more than 3 distinct answers 
            if 0:
                if len(row_dict['doc_predictions']) > 3:
                    temp_doc_predictions, temp_doc_scores = [],[]
                    for i in range(len(row_dict['doc_predictions'])):
                        if row_dict['doc_predictions'][i] in row_dict['answers']:
                            if random.uniform(0, 1) < 0.3:
                                pass
                                #don't save this answer
                            else:
                                temp_doc_predictions.append(row_dict['doc_predictions'][i])
                                temp_doc_scores.append(row_dict['doc_scores'][i])
                        else:
                            temp_doc_predictions.append(row_dict['doc_predictions'][i])
                            temp_doc_scores.append(row_dict['doc_scores'][i])
                    row_dict['doc_predictions'] = temp_doc_predictions
                    row_dict['doc_scores'] = temp_doc_scores
                            

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
    print(exceed_counter,' instances of possible wrong candidate segmentation')
    print('recall',positive_counter/float(total_counter),'hit rate',scores_counter/total_counter)
    with open(json_file_path, 'w') as json_file_handler:
        #Step 4
        json_file_handler.write(json.dumps(data_dict, indent = 4))
 

csv_file_path = "/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/testing_results/RAVQA-InstructBLIP-test-K50.csv"
json_file_path = "/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/testing_results/RAVQA-InstructBLIP-test-K50.json"
blip2_caption_path = "/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/pre-extracted_features/captions/coco_blip2_captions_val2014.json"
promptcap_caption_path = "/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/pre-extracted_features/captions/coco_promptcap_captions_val2014.json"
with open(blip2_caption_path) as json_file:
    blip2_captions = json.load(json_file)
with open(promptcap_caption_path) as json_file:
    promptcap_captions = json.load(json_file)
#print(blip2_captions)
csv_to_json(csv_file_path, json_file_path, K=50)

csv_file_path = "/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/testing_results/RAVQA-InstructBLIP-train-K50.csv"
json_file_path = "/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/testing_results/RAVQA-InstructBLIP-train-K50.json"
blip2_caption_path = "/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/pre-extracted_features/captions/coco_blip2_captions_train2014.json"
promptcap_caption_path = "/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/pre-extracted_features/captions/coco_promptcap_captions_train2014.json"
with open(blip2_caption_path) as json_file:
    blip2_captions = json.load(json_file)
with open(promptcap_caption_path) as json_file:
    promptcap_captions = json.load(json_file)
csv_to_json(csv_file_path, json_file_path, K=50)

#take flan-t5-xxl csv result file from wandb or json file from gpt to evluated result
import csv
import json
import re

def get_score(answers,prediction):
    def processPunctuation(inText):
        punct        = [';', r"/", '[', ']', '"', '{', '}',
                             '(', ')', '=', '+', '\\', '_', '-',
                             '>', '<', '@', '`', ',', '?', '!']
        periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
        commaStrip   = re.compile("(\d)(,)(\d)")
        outText = inText
        for p in punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = periodStrip.sub("",outText,re.UNICODE)
        return outText
    def processDigitArticle(inText):
        articles     = ['a','an','the']
        manualMap    = { 'none': '0',
                              'zero': '0',
                              'one': '1',
                              'two': '2',
                              'three': '3',
                              'four': '4',
                              'five': '5',
                              'six': '6',
                              'seven': '7',
                              'eight': '8',
                              'nine': '9',
                              'ten': '10'}
        contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
        "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
                             "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've",
                             "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've",
                             "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
                             "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
                             "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
                             "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
                             "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
                             "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
                             "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
                             "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
                             "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've",
                             "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
                             "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
                             "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",
                             "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've",
                             "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
                             "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
                             "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
                             "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've",
                             "youll": "you'll", "youre": "you're", "youve": "you've"}
        
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = manualMap.setdefault(word, word)
            if word not in articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in contractions: 
                outText[wordId] = contractions[word]
        outText = ' '.join(outText)
        return outText
    prediction = processPunctuation(prediction)
    prediction = processDigitArticle(prediction)
    no=answers.count(prediction)
    
    if no>3:
        soft_score=1
    elif no==3:
        soft_score=0.9
    elif no==2:
        soft_score=0.6
    elif no==1:
        soft_score=0.3
    else:
        soft_score=0
    '''
    
    soft_score=no/3.0
    
    if soft_score>1:
        soft_score=1
    '''
    return soft_score

def csv_result(csv_result_file_path):
    in_scores_list, out_scores_list=[],[]
    in_count, out_count=0,0
    with open(csv_result_file_path) as csv_file_handler:
        csv_reader = csv.DictReader(csv_file_handler)
        for rows in csv_reader:
            answers=rows['answers'].split(',')
            prediction=rows['prediction']
            prompt=rows['prompt']
            
            #since this model repeat the whole prompt, only get last answer
            start=[m.start() for m in re.finditer('Candidates: ', prompt)][-1]
            end=[m.start() for m in re.finditer('Answer:', prompt)][-1]
            prompt=prompt[start+12:end-3].split(',')
            
            answer_candidates=[]
            for ac in prompt:
                bracket_index = ac.find('(')
                answer_candidates.append(ac[:bracket_index].strip())
            #print(answer_candidates)
            score=get_score(answers,prediction)
            if prediction in answer_candidates:
                in_count+=1
                in_scores_list.append(score)
            else:
                out_count+=1
                out_scores_list.append(score)
    
    print('Prediction in AC to prediction not in AC ratio is',round(in_count/float(in_count+out_count),4))
    print('Each type score is',round(sum(in_scores_list)/in_count*100, 2),round(sum(out_scores_list)/out_count*100,2))
    print('VQA Score',(sum(in_scores_list)+sum(out_scores_list))/(in_count+out_count))
   
def json_result(json_result_file_path):
    f = open(json_result_file_path)
    data = json.load(f)
    f = open("/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/testing_results/RAVQA-InstructBLIP-test-K50.json")
    annotation = json.load(f)
    in_scores_list, out_scores_list=[],[]
    in_count, out_count=0,0
    for question_id in data:
        answers=annotation[question_id]['answers']
        prediction=data[question_id]['answer']
        #post process prediction to remove brackets and keep first answer only
        bracket_index = prediction.find('(')
        if bracket_index == -1:
            prediction=prediction.strip()
        else:
            prediction = prediction[:bracket_index].strip()
        prompt=data[question_id]['prompt_info'][0]['prompt']
        #print(prompt)
        
        #since this model repeat the whole prompt, only get last answer
        start=[m.start() for m in re.finditer('Candidates: ', prompt)][-1]
        end=[m.start() for m in re.finditer('Answer:', prompt)][-1]
        prompt=prompt[start+12:end-3].split(',')
        
        answer_candidates=[]
        for ac in prompt:
            bracket_index = ac.find('(')
            answer_candidates.append(ac[:bracket_index].strip())
        #print(answer_candidates)
        score=get_score(answers,prediction)
        if prediction in answer_candidates:
            in_count+=1
            in_scores_list.append(score)
        else:
            out_count+=1
            out_scores_list.append(score)
    print(in_count,out_count)
    print('Prediction in AC to prediction not in AC ratio is',round(in_count/float(in_count+out_count),4))
    print('Each type score is',round(sum(in_scores_list)/in_count*100, 2),round(sum(out_scores_list)/out_count*100,2))
    print('VQA Score',(sum(in_scores_list)+sum(out_scores_list))/(in_count+out_count))
         

csv_result_file_path = "/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/prophet-main/data/few_shot_final_1.csv"
csv_result(csv_result_file_path)

#json_result_file_path = "/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/prophet-main/outputs/successful_runs/cache_20230911010141.json"
#json_result(json_result_file_path)
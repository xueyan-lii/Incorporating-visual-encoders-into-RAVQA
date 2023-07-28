# get faiss index for instructBLIP image encoding and question
# use MLMI8 environment
import pickle
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import faiss
import numpy as np
import random
import json

data_dir = Path("/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa")
#vqa2_data_dir = data_dir / "vqa2"

USE_GPU = False
RERUN_INDEX = True
D_FILEPATH = '/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/cache/concat_nearest_neightbours_distance_20.npy'
I_FILEPATH = '/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/cache/concat_nearest_neighbours_index_20.npy' 
OUT_PATH = data_dir / f"pre-extracted_features/in_context_examples/rices_concat_a1b1_normalized_text_oscar_caption.pkl"
ALPHA = 1.0 # weight on text embeddings
BETA = 0.0 # weight on image embeddings
TOP_K = 20 # number of neighbours to retrieve


def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        load_pickle_data = pickle.load(f)
    return load_pickle_data

def dict_data_to_df(id_embed_dict, embed_field_name='text_embedding'):
    if embed_field_name == 'text_embedding':
        id_field_name = 'question_id'
    elif embed_field_name == 'image_embedding':
        id_field_name = 'question_id' #also use question_id due to instructblip embedding used
    tmp_dict = {id_field_name: [], embed_field_name: []}
    for _id, embed in id_embed_dict.items():
        tmp_dict[id_field_name].append(str(_id))
        tmp_dict[embed_field_name].append(embed)
    res_df = pd.DataFrame(tmp_dict)
    #if embed_field_name == 'text_embedding':
    #    res_df['question_id_prefix'] = res_df['question_id'].apply(lambda x: str(x)[:-3])
    return res_df

def concat_embeddings_with_weights(df, embed_field_1, embed_field_2, alpha=1.0, beta=1.0):
    #TODO Exercise 4.2: Complete the function concat_embeddings_with_weights:
    # Args:
    #   df: A df.DataFrame object with fields for embeddings
    #   embed_field_1: field name for the first embedding
    #   embed_field_2: field name for the second embedding
    #   alpha: weighting for the first embedding
    #   beta: weighting for the second embedding
    # Returns:
    #   A numpy matrix where the embeddings from df[embed_field_1] and df[embed_field_1]  
    #   are concatanented row-wise. The embeddings should be weighted by alpha and beta, respectively
    #   i.e., The i-th row of the return matrix should be a np.array that contains the weighted concatenation of df.iloc[i][field_1] and df.iloc[i][field_2]
    # Hint:
    #   Use np.concatenate to concat the embeddings. See documentation here: https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
    #   You can change a dataframe column to an np.array by calling `df[col_name].values`. You may also find `np.stack()`, `np.squeeze()` and `DataFrame.iterrows()` useful.
    #   Use `faiss.normalize_L2(embeddings)`` to normalize vectors if you need to. What should you normalize? 
    result = None
    ##### Exercise 4.2 BEGIN ##### 
    crib_dict = {"df": df, "embed_field_1": embed_field_1, "embed_field_2": embed_field_2, "alpha": alpha, "beta": beta}
    #print('dfx\n',df)
    result= []
    #
    temp1 = np.concatenate(df[embed_field_1])
    faiss.normalize_L2(temp1)
    temp2 = np.concatenate(df[embed_field_2])
    faiss.normalize_L2(temp2)

    for i in range(len(temp1)):
        result.append(np.concatenate((np.dot(temp1[i], alpha), np.dot(temp2[i], beta))))
    result=np.array(result)
    #print('results\n',result)
    result = crib_dict.get('result', result) 
    return result
    ##### Exercise 4.2 END ##### 

def row_index_to_question_id(df, r_ind):
    return df.iloc[r_ind]['question_id']

def load_preprocessed_data(file_path):
    with open(file_path, 'rb') as f:
        load_pickle_data = pickle.load(f)["cache"]
    data_dict = {
        str(item['question_id']):item
        for item in load_pickle_data['data_items']
    }
    return data_dict 

if __name__ == '__main__':
    val_text_embeddings_df = dict_data_to_df(load_pkl(data_dir / "pre-extracted_features/text_embeddings/InstructBLIP_question+oscar_caption_embeddings_val2014.pkl"))
    val_text_embeddings_df['text_embedding'] = val_text_embeddings_df.apply(lambda row : np.expand_dims(np.array(row['text_embedding'].flatten()) ,axis=0), axis=1)
    val_image_embeddings_df = dict_data_to_df(load_pkl(data_dir / "pre-extracted_features/text_embeddings/InstructBLIP_image_embeddings_val2014.pkl"), embed_field_name='image_embedding')
    val_image_embeddings_df['image_embedding'] = val_image_embeddings_df.apply(lambda row : np.expand_dims(np.array(row['image_embedding'].flatten()) ,axis=0), axis=1)
    train_text_embeddings_df = dict_data_to_df(load_pkl(data_dir / "pre-extracted_features/text_embeddings/InstructBLIP_question+oscar_caption_embeddings_train2014.pkl"))
    train_text_embeddings_df['text_embedding'] = train_text_embeddings_df.apply(lambda row : np.expand_dims(np.array(row['text_embedding'].flatten()) ,axis=0), axis=1)
    train_image_embeddings_df = dict_data_to_df(load_pkl(data_dir / "pre-extracted_features/text_embeddings/InstructBLIP_image_embeddings_train2014.pkl"), embed_field_name='image_embedding')
    train_image_embeddings_df['image_embedding'] = train_image_embeddings_df.apply(lambda row : np.expand_dims(np.array(row['image_embedding'].flatten()) ,axis=0), axis=1)

    val_df = None
    train_df = None
    #TODO Exercise 4.1: Produce a merged dataframe that contains image and text embeddings. Do this for training and validation data
    # DataFrame merge documentation: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html 
    # Hint: consider what field should be used to merge the text and image DataFrames to set the `on` parameter of dataframe.merge(). 
    # You also need to decide the correct type of merge (left, right, cross) and specify the `how` parameter of dataframe.merge(). Note that we want to obtain concatenated embeddings for each **questions**.
    ##### Exercise 4.1 BEGIN ##### 
    crib_dict = {}
    
    #print('val_text_embeddings_dfx \n', val_text_embeddings_df)# shape 1x768
    #print('val_image_embeddings_dfx \n', val_image_embeddings_df)# shape 1x768

    val_df = val_text_embeddings_df.merge(val_image_embeddings_df, how='left', on='question_id')
    train_df = train_text_embeddings_df.merge(train_image_embeddings_df, how='left', on='question_id')
    #print('val_dfx\n',val_df)
    val_df = crib_dict.get('val_df', val_df)
    train_df = crib_dict.get('train_df', train_df)
    ##### Exercise 4.1 BEGIN ##### 
    val_concat_embeddings_mat = concat_embeddings_with_weights(val_df, 'text_embedding', 'image_embedding', alpha=ALPHA, beta=BETA)
    train_concat_embeddings_mat = concat_embeddings_with_weights(train_df, 'text_embedding', 'image_embedding', alpha=ALPHA, beta=BETA)
    print(train_concat_embeddings_mat.shape[1]) #32x2048x2=131072
    D, I = None, None
    if RERUN_INDEX:
        index = faiss.IndexFlatIP(train_concat_embeddings_mat.shape[1])
        if USE_GPU:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print(index.is_trained)
        index.add(train_concat_embeddings_mat)

        print(f"Embeddings in the index file: {index.ntotal}")
        #search for similar paris from training set
        D, I = index.search(val_concat_embeddings_mat, TOP_K)
        with open(I_FILEPATH, 'wb') as f:
            np.save(f, I)
        with open(D_FILEPATH, 'wb') as f:
            np.save(f, D)
    else:
        with open(I_FILEPATH, 'rb') as f:
            I = np.load(f)
        with open(D_FILEPATH, 'rb') as f:
            D = np.load(f)

    val_df['nn_question_ids'] = [row_index_to_question_id(train_df, row_ind) for row_ind in tqdm(I, desc='assigning nn_question_ids')]
    val_df['nn_similarities'] = [s for s in tqdm(D, desc='unpacking similarities')]
    # load preprocessed datasets only for img_path
    train_data_vqa2 = load_preprocessed_data(data_dir / "cache/train_data_preprocessed.pkl")
    val_data_vqa2 = load_preprocessed_data(data_dir / "cache/test_data_preprocessed.pkl")
    #load the clean answer candidates
    f = open('/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/testing_results/RAVQA-InstructBLIP-train-K50.json')
    train_predictions = json.load(f)
    f = open('/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/testing_results/RAVQA-InstructBLIP-test-K50.json')
    val_predictions = json.load(f)


    in_context_examples_dict = {}
    for i, row in tqdm(val_df.iterrows(), desc='Generating in_context_examples'):
        val_question_id = row['question_id']
        
        in_context_examples_dict[val_question_id] \
            = sorted([ 
                dict(
                    **train_predictions[nn_question_id],
                    similarity=nn_sim,
                    val_question=val_data_vqa2[val_question_id]['question'],
                    val_image_key=val_data_vqa2[val_question_id]['img_key'],
                    val_image_path=val_data_vqa2[val_question_id]['img_path'],
                    val_doc_scores=val_predictions[val_question_id]['doc_scores'],
                    val_doc_predictions=val_predictions[val_question_id]['doc_predictions'],
                    val_oscar_caption=val_predictions[val_question_id]['oscar_caption'],
                    val_blip2_caption=val_predictions[val_question_id]['blip2_caption'],
                )
                for nn_question_id, nn_sim in zip(row['nn_question_ids'], row['nn_similarities'])
              ], key=lambda x: x['similarity'], reverse=True)
        '''
        print('new row')
        for j in in_context_examples_dict[val_question_id]:
            print(j['val_question'])
            print(j['question'])
            print(j['similarity'])
        '''

    with open(OUT_PATH, "wb") as f:
        pickle.dump(in_context_examples_dict, f)

#%%
# Import require dpackages, ignore warnings
import warnings
warnings.filterwarnings('ignore')
import fitz
import re
import pandas as pd
import numpy as np
# import tensorflow
import torch
import collections
import os 
from operator import itemgetter
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import LayoutLMv2FeatureExtractor

#%%
##################---------LAYOUT SEGMENTATION------------##########################
#-----------------------------------------------------------------------------------
# A. KEY PARAMETERS & MODELS
#--------------------------------
# Import the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("pierreguillou/lilt-xlm-roberta-base-finetuned-with-DocLayNet-base-at-paragraphlevel-ml512")
model = AutoModelForTokenClassification.from_pretrained("pierreguillou/lilt-xlm-roberta-base-finetuned-with-DocLayNet-base-at-paragraphlevel-ml512")
feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
# get labels
id2label = model.config.id2label
label2id = model.config.label2id
num_labels = len(id2label)

# bounding boxes (bbox) start and end of a sequence
cls_box = [0, 0, 0, 0]
sep_box = cls_box

# parameters for tokenization and overlap
max_length = 512 # The maximum length of a feature (sequence)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
# %%
# B. GENERAL FUNCTIONS
# ------------------------

# Function to extract text and bbox from the document
def extract_text_bbox(page_ranges, doc):
    """input params: 
                - page_ranges: ranges of pages to be read from the document
                - doc: fitz document
    output params: 
                - pages: fitz pages """
    
    df = pd.DataFrame()

    height = []
    width = []
    data = []

    if page_ranges == [(0,0)]:
        page_ranges =[(0,1)]
    for page_range in page_ranges:
        start, end = page_range
        start = start - 1

        for page in doc.pages():
            if page.number in range(start, end):
                height.append(page.mediabox[3])
                width.append(page.mediabox[2])
                data.append(page.get_text("blocks"))
        
    pages = dict({"height":height , "width":width , "data":data})
                # pages.append(page.get_text("blocks"))
             
    return pages

# Function to create dataset out of the extracted data
def text_bbox_dataset(pages):
    """ From the extracted pages, this function create a dataset out of it,
    and returns as a datasets.Dataset 
    """
    num_pages = len(pages['height'])
    texts,bboxes,page_num,number_of_pages,height,width = list(),list(), list(),list(),list(), list()
    if num_pages > 0:
        try:
            #iterate throguh pages (1 page is pne row of dataset)
            for p,page in enumerate(pages['data']):
                p_texts = []
                p_bboxes = []
                page_num.append(p)
                number_of_pages.append(num_pages)
                height.append(pages['height'][p])
                width.append(pages['width'][p])


                #iterate throguh blocks 
                for b, block in enumerate(page):
                    text = block[4]
                    if not text.isspace():
                        p_texts.append(re.sub('\s*\n\s*','',text))
                        p_bboxes.append([block[0],block[1],block[2],block[3]])
                texts.append(p_texts)
                bboxes.append(p_bboxes)
        except:
            print(f"There was an error within the extraction of PDF text")
        else:
            from datasets import Dataset
            dataset = Dataset.from_dict({"page_num":page_num, "num_pages": number_of_pages, "texts":texts,"bboxes":bboxes,
                                        "height": height, "width": width})
            return dataset

        
# Function to ensure the format of bbox
def upperleft_to_lowerright(bbox):
    x0, y0, x1, y1 = tuple(bbox)
    if bbox[2] < bbox[0]:
        x0 = bbox[2]
        x1 = bbox[0] 
    if bbox[3] < bbox[1]:
        y0 = bbox[3]
        y1 = bbox[1] 
    return [x0, y0, x1, y1]

# LiLT model gets 1000x10000 pixels images
def normalize_box(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

# LiLT model gets 1000x10000 pixels images
def denormalize_box(bbox, width, height):
    return [
        int(width * (bbox[0] / 1000)),
        int(height * (bbox[1] / 1000)),
        int(width* (bbox[2] / 1000)),
        int(height * (bbox[3] / 1000)),
    ]
# function to sort bounding boxes
def get_sorted_boxes(bboxes):
    # sort by y from page top to bottom 
    sorted_bboxes = sorted(bboxes, key=itemgetter(1), reverse=False)
    y_list = [bbox[1] for bbox in sorted_bboxes]

    # sort by x from page left to right when boxes with same y
    if len(list(set(y_list))) != len(y_list):
        y_list_duplicates_indexes = dict()
        y_list_duplicates = [item for item, count in collections.Counter(y_list).items() if count > 1]
        for item in y_list_duplicates:
            y_list_duplicates_indexes[item] = [i for i, e in enumerate(y_list) if e == item]
            bbox_list_y_duplicates = sorted(np.array(sorted_bboxes, dtype=object)[y_list_duplicates_indexes[item]].tolist(), key=itemgetter(0), reverse=False)
            np_array_bboxes = np.array(sorted_bboxes)
            np_array_bboxes[y_list_duplicates_indexes[item]] = np.array(bbox_list_y_duplicates)
            sorted_bboxes = np_array_bboxes.tolist()

    return sorted_bboxes

# sort data without labels
def sort_data_wo_labels(bboxes, texts):

    sorted_bboxes = get_sorted_boxes(bboxes)
    sorted_bboxes_indexes = [bboxes.index(bbox) for bbox in sorted_bboxes]
    sorted_texts = np.array(texts, dtype=object)[sorted_bboxes_indexes].tolist()

    return sorted_bboxes, sorted_texts

# C. FUNCTION/S TO PREPARE FEATURES MODEL READY
# ------------------------
# Prepare the features ready for using with the model
def prepare_features(example,cls_box = cls_box, sep_box = sep_box):
#     width = 595.32
#     height = 841.92
    max_length = 512
    doc_stride = 128
    pages_ids_list, chunks_ids_list, input_ids_list, attention_mask_list, bb_list = list(), list(), list(), list(), list()
    
    # get batch
    batch_pages_ids = example["page_num"]
    # batch_pages = example["pages"]
    batch_bboxes_par = example["bboxes"]
    batch_texts_par = example["texts"]  
    # batch_pages_size = [page.size for image in batch_pages]
#     batch_width, batch_height = [width for i in range(len(batch_texts_par))],[height for i in range(len(batch_texts_par))]
    batch_width = example["width"]
    batch_height = example["height"]
    # add a dimension if not a batch but only one image
    if not isinstance(batch_pages_ids, list): 
        batch_pages_ids = [batch_pages_ids]
        batch_pages = [batch_pages]
        batch_bboxes_par = [batch_bboxes_par]
        batch_texts_par = [batch_texts_par]
        batch_width, batch_height = [batch_width], [batch_height]
        
    # process all images of the batch
    for num_batch, (page_id, boxes, texts_par, width, height) in enumerate(zip(batch_pages_ids, batch_bboxes_par, batch_texts_par, batch_width, batch_height)):
        tokens_list = []
        bboxes_list = []
        
        # add a dimension if only on image
        if not isinstance(texts_par, list):
            texts_par, boxes = [texts_par], [boxes]
            
        # normalize bboxes
        normalize_bboxes_par = [normalize_box(upperleft_to_lowerright(box), width, height) for box in boxes]
        
#         print(normalize_bboxes_par)
        
        # sort boxes with texts
        # we want sorted lists from top to bottom of the image
        boxes, texts_par = sort_data_wo_labels(normalize_bboxes_par, texts_par)
        
        count = 0
        for box, text_par in zip(boxes, texts_par):
            tokens_par = tokenizer.tokenize(text_par)
            num_tokens_par = len(tokens_par) # get number of tokens
            tokens_list.extend(tokens_par)
            bboxes_list.extend([box] * num_tokens_par) # number of boxes must be the same as the number of tokens
            
        # use of return_overflowing_tokens=True / stride=doc_stride
        # to get parts of image with overlap
        # source: https://huggingface.co/course/chapter6/3b?fw=tf#handling-long-contexts
        encodings = tokenizer(" ".join(texts_par), 
                              truncation=True,
                              padding="max_length", 
                              max_length=max_length, 
                              stride=doc_stride, 
                              return_overflowing_tokens=True, 
                              return_offsets_mapping=True
                              )

        otsm = encodings.pop("overflow_to_sample_mapping")
        offset_mapping = encodings.pop("offset_mapping")
        
        # Let's label those examples and get their boxes   
        sequence_length_prev = 0   
        for i, offsets in enumerate(offset_mapping):
            # truncate tokens, boxes and labels based on length of chunk - 2 (special tokens <s> and </s>)
            sequence_length = len(encodings.input_ids[i]) - 2
            if i == 0: start = 0
            else: start += sequence_length_prev - doc_stride
            end = start + sequence_length
            sequence_length_prev = sequence_length

            # get tokens, boxes and labels of this image chunk
            bb = [cls_box] + bboxes_list[start:end] + [sep_box]

            # as the last chunk can have a length < max_length
            # we must to add [tokenizer.pad_token] (tokens), [sep_box] (boxes) and [-100] (labels)
            if len(bb) < max_length:
                bb = bb + [sep_box] * (max_length - len(bb))

            # append results
            input_ids_list.append(encodings["input_ids"][i])
            attention_mask_list.append(encodings["attention_mask"][i])
            bb_list.append(bb)
            pages_ids_list.append(page_id)
            chunks_ids_list.append(i)
            
    return {
      "pages_ids": pages_ids_list,
      "chunk_ids": chunks_ids_list,
      "input_ids": input_ids_list,
      "attention_mask": attention_mask_list,
      "normalized_bboxes": bb_list,
  }
# Creating custom dataset (make it easy to work with huggingface)
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
    # get item
        example = self.dataset[idx]
        encoding = dict()
        encoding["chunk_ids"] = example["chunk_ids"]
        encoding["input_ids"] = example["input_ids"]
        encoding["attention_mask"] = example["attention_mask"]
        encoding["bbox"] = example["normalized_bboxes"]
        encoding["pages_ids"] = example["pages_ids"]
        
        return encoding

#--------------------------------------------------------------------------------------
# D. MODELS AND PREDICTION FUNCTIONS
# -------------------------------------------------------------------------------------    
model = model.to('cuda')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import torch.nn.functional as F

# Token level
def predictions_token_level_gpu(pages, custom_encoded_dataset):

    num_pgs = len(pages)
    if num_pgs > 0:

        chunk_ids, input_ids, bboxes, outputs, token_predictions  = dict(), dict(), dict(), dict(), dict()
        normalize_batch_bboxes_lines_pars = dict()
        pages_ids_list = list()

        for i,encoding in enumerate(custom_encoded_dataset):

            # get custom encoded data
            page_id = encoding['pages_ids']
            chunk_id = encoding['chunk_ids']
            input_id = torch.tensor(encoding['input_ids'])[None]
            attention_mask = torch.tensor(encoding['attention_mask'])[None]
            bbox = torch.tensor(encoding['bbox'])[None]
            
            # save data in dictionnaries
            if page_id not in pages_ids_list: pages_ids_list.append(page_id)

            if page_id in chunk_ids: chunk_ids[page_id].append(chunk_id)
            else: chunk_ids[page_id] = [chunk_id]

            if page_id in input_ids: input_ids[page_id].append(input_id)
            else: input_ids[page_id] = [input_id]

            if page_id in bboxes: bboxes[page_id].append(bbox)
            else: bboxes[page_id] = [bbox]
            
#             input_id = input_id.to('cuda')
#             attention_mask = attention_mask.to('cuda')
#             bbox= bbox.to('cuda')
            # convert device of parameters into cuda
            with torch.autograd.detect_anomaly():
                input_id = input_id.to('cuda')
                attention_mask = attention_mask.to('cuda')
                bbox= bbox.to('cuda')

            # get prediction with forward pass
            with torch.no_grad():
                output = model(
                    input_ids=input_id,
                    attention_mask=attention_mask,
                    bbox=bbox
                    )

            # save probabilities of predictions in dictionnary
            if page_id in outputs: outputs[page_id].append(F.softmax(output.logits.squeeze(), dim=-1))
            else: outputs[page_id] = [F.softmax(output.logits.squeeze(), dim=-1)]

        return outputs, pages_ids_list, chunk_ids, input_ids, bboxes

    else:
        print("An error occurred while getting predictions!")

# final result dataframe creation
# dataframe level
from functools import reduce

def prediction_dataframe_gpu(dataset,outputs,pages, chunks,inputs,bboxes):#incorporate width, height with page, and reduce parameters later
    ten_probs_dict, ten_input_ids_dict, ten_bboxes_dict = dict(), dict(), dict()
    bboxes_list_dict, input_ids_dict_dict, probs_dict_dict, df = dict(), dict(), dict(), dict()
    
    if len(pages) > 0:
        for i,page_id in enumerate(pages):
            
            # get page metadata
            page_list = dataset.filter(lambda example: example["page_num"] == page_id)["page_num"]
            page = page_list[0]
            width = dataset["width"][i]
            height = dataset["height"][i]
            
            # get data
            chunk_ids_list = chunks[page_id]
            outputs_list = outputs[page_id]
#             print(outputs[1])
            input_ids_list = inputs[page_id]
            bboxes_list = bboxes[page_id]
            
            # create zeros tensors
            ten_probs = torch.zeros((outputs_list[0].shape[0] - 2)*len(outputs_list), outputs_list[0].shape[1])
            ten_input_ids = torch.ones(size=(1, (outputs_list[0].shape[0] - 2)*len(outputs_list)), dtype =int)
            ten_bboxes = torch.zeros(size=(1, (outputs_list[0].shape[0] - 2)*len(outputs_list), 4), dtype =int)
            
            ten_probs = ten_probs.to('cuda')
            ten_input_ids = ten_input_ids.to('cuda')
            ten_bboxes = ten_bboxes.to('cuda')
            
            if len(outputs_list) > 1:
              
                for num_output, (output, input_id, bbox) in enumerate(zip(outputs_list, input_ids_list, bboxes_list)):
                    start = num_output*(max_length - 2) - max(0,num_output)*doc_stride
                    end = start + (max_length - 2)
                    
                    if num_output == 0:
                        ten_probs[start:end,:] += output[1:-1]
                        ten_input_ids[:,start:end] = input_id[:,1:-1]
                        ten_bboxes[:,start:end,:] = bbox[:,1:-1,:]
                    else:
                        ten_probs[start:start + doc_stride,:] += output[1:1 + doc_stride]
                        ten_probs[start:start + doc_stride,:] = ten_probs[start:start + doc_stride,:] * 0.5
                        ten_probs[start + doc_stride:end,:] += output[1 + doc_stride:-1]

                        ten_input_ids[:,start:start + doc_stride] = input_id[:,1:1 + doc_stride]
                        ten_input_ids[:,start + doc_stride:end] = input_id[:,1 + doc_stride:-1]

                        ten_bboxes[:,start:start + doc_stride,:] = bbox[:,1:1 + doc_stride,:]
                        ten_bboxes[:,start + doc_stride:end,:] = bbox[:,1 + doc_stride:-1,:]
              
            else:
                ten_probs += outputs_list[0][1:-1] 
                ten_input_ids = input_ids_list[0][:,1:-1] 
                ten_bboxes = bboxes_list[0][:,1:-1] 
                
            ten_probs_list, ten_input_ids_list, ten_bboxes_list = ten_probs.tolist(), ten_input_ids.tolist()[0], ten_bboxes.tolist()[0]
            bboxes_list = list()
            input_ids_dict, probs_dict = dict(), dict()
            bbox_prev = [-100, -100, -100, -100]
            for probs, input_id, bbox in zip(ten_probs_list, ten_input_ids_list, ten_bboxes_list):
#                 bbox = denormalize_box(bbox, width, height)
                if bbox != bbox_prev and bbox != cls_box and bbox != sep_box and bbox[0] != bbox[2] and bbox[1] != bbox[3]:
                    bboxes_list.append(bbox)
                    input_ids_dict[str(bbox)] = [input_id]
                    probs_dict[str(bbox)] = [probs]
                elif bbox != cls_box and bbox != sep_box and bbox[0] != bbox[2] and bbox[1] != bbox[3]:
                    input_ids_dict[str(bbox)].append(input_id)
                    probs_dict[str(bbox)].append(probs)
                bbox_prev = bbox
                
            probs_bbox = dict()
            for i,bbox in enumerate(bboxes_list):
                probs = probs_dict[str(bbox)]
                probs = np.array(probs).T.tolist()
            
                probs_label = list()
                for probs_list in probs:
                    prob_label = reduce(lambda x, y: x*y, probs_list)
                    prob_label  = prob_label**(1./(len(probs_list))) # normalization
                    probs_label.append(prob_label)
                max_value = max(probs_label)
                max_index = probs_label.index(max_value)
                probs_bbox[str(bbox)] = max_index

            bboxes_list_dict[page_id] = bboxes_list
            input_ids_dict_dict[page_id] = input_ids_dict
            probs_dict_dict[page_id] = probs_bbox

            df[page_id] = pd.DataFrame()
            df[page_id]["bboxes"] = bboxes_list
            df[page_id]["texts"] = [tokenizer.decode(input_ids_dict[str(bbox)]) for bbox in bboxes_list]
            df[page_id]["labels"] = [id2label[probs_bbox[str(bbox)]] for bbox in bboxes_list]

        return probs_bbox, bboxes_list_dict, input_ids_dict_dict, probs_dict_dict, df

    else:
        print("An error occurred while getting predictions!")

# %%

#############################################
# LAYOUT SEGMENTATION MODEL PIPELINE, FROM THE FILE READING TO THE PREDICTION -------------------------
# -------------------------------------------------------------------------------------
def layoutSegmentation(file_name, page_ranges= None):


    doc = fitz.open(file_name)
    if page_ranges == None:
        page_ranges = [(0,len(doc)-1)] #if not mentioned the ranges, entire document is considered

    # Extract data
    pages_dict = extract_text_bbox(page_ranges, doc)
    data = text_bbox_dataset(pages_dict)

    # Encoded dataset
    encoded_ = data.map(prepare_features, batched =True, batch_size = 64, remove_columns = data.column_names)
    # custom encoded dataset
    custom_ = CustomDataset(encoded_, tokenizer)
    pages_ = set(encoded_['pages_ids'])
    # getting predictions
    outputs_, pages_, chunks_,inputs_, bboxes_ = predictions_token_level_gpu(pages_, custom_)
    # final out
    bbox,_,_,_,df_ = prediction_dataframe_gpu(data, outputs_,pages_,chunks_, inputs_, bboxes_)

    return df_


# %%
####################------------------DOC PARSER - AFTER LAYOUT SEGMENTATION - GETTING THE STRUCTURED DATAFRAME-------------#######################################
def postProcessor(df_list):
    
    paras = ['']
    sections = ['']
    titles = ['']
    list_items = [0]
    page_num = [0]
    # bbox = []
    # Iterate through the df list to get df
    for i,df in enumerate(df_list.values()):
        for r,row in df.iterrows():
            try:
                if row.labels == 'Title':
                #    text = fix_spelling(row.texts)
                #    titles.append(text)
                   titles.append(row.texts)
                   sections.append('')
                   paras.append('')
                   page_num.append(i)
                #    bbox.append(row.bbox)
                elif row.labels == 'Section-header':
                    # text = fix_spelling(row.texts)
                    sections.append(row.texts)
                    # sections.append(text)
                    titles.append(titles[-1])
                    paras.append('')
                    page_num.append(i)
                elif (row.labels == 'Text' or row.labels == 'List-item') and not re.match('\<image\:.+>',row.texts):
                    if row.labels == 'List-item':
                        row.labels.append(list_items)
                        if df['labels'][r+1] != 'List-item':
                            paras.append(', '.join(list_items)) # to treat the list items properly, without lose the meaning.
                            page_num.append(i)
                            list_items = []
                    else:
                        # text = fix_spelling(row.texts)
                        # paras.append(text)
                        paras.append(row.texts)
                        sections.append(sections[-1])
                        titles.append(titles[-1])
                        page_num.append(i)
            except:
                pass

    df = pd.DataFrame()
    df['Title'] = titles
    df['Section'] = sections
    df['Content'] = paras 
    df['Page'] = page_num
    df.drop(0, inplace=True)
    return df

# %%
# COMPLETE PIPELINE - CALLING THE LAYOUT SEGMENTATION AND POST PROCESSOR
def docParser(file_name, page_ranges = None):
    """file_name: string, file location and name,
    page_ranges: list of tuples showing ranges, eg:- [(1,123),(300,350)]
    returns structured dataframe of the document
    """
    segmented_pages = layoutSegmentation(file_name, page_ranges)
    structured_df = postProcessor(segmented_pages)

    return structured_df
        

# %%

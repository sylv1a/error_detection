from functions import (
    CustomLLM,
    load_prompt,
    create_output,
    load_data,
)  # ,load_input if used
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.schema import OutputParserException
import time
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings('always')

config_path="/home/schacha@FuL.med.uni-muenchen.de/project/llm-pathology-extraction/config/config.yaml"
error_detection_prompt="/home/schacha@FuL.med.uni-muenchen.de/project/llm-pathology-extraction/src/baseline/prompts/baseline_error_prompt.txt"
error_detection_prompt2="/home/schacha@FuL.med.uni-muenchen.de/project/llm-pathology-extraction/src/errors/baseline/propmts/baseline_report.txt"

llm = CustomLLM()
llm.load_config(config_path)

parser = JsonOutputParser()
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

data_path = "/home/schacha@FuL.med.uni-muenchen.de/project/llm-pathology-extraction/llm-reports/last_25/Approach2_100_more.csv"
data=load_data(data_path)
data

prompt = PromptTemplate(
    template="{prompt1}{input1}{prompt2}",
    input_variables=["input1"],
    partial_variables={
        "prompt1": load_prompt(error_detection_prompt),
        "prompt2": load_prompt(error_detection_prompt2)
    },
)

for k in range(len(data)):
    start = time.time()
    print(k)
    report_index=data["report_index"][k]
    input=data["report_text"][k]
    final_data = pd.DataFrame()

    try:
        error_prediction = []
        error_type_predictions = []
        error_positions = []
        chain = prompt | llm 
        response = chain.invoke(
           {"input1": input}
        )
        response_output=json.loads(response.split("<output>")[1].split("</output>")[0])
        
        final_data.at[k,"report_index"]=report_index
        final_data.at[k,"report"]=input
        final_data.at[k,"errors"]=response_output
        for i in response_output:
            error_prediction.append(i["error_label"])
            error_type_predictions.append(i["error_type_label"]) 
            if i["findings_sentence"]!="" and i["impression_sentence"]!="":
                pos=data["report_text"][k].find(i["findings_sentence"])
                error_positions.append(pos)
                pos=data["report_text"][k].find(i["impression_sentence"])
                error_positions.append(pos)
            else: 
                if i["findings_sentence"]!="":
                    pos=data["report_text"][k].find(i["findings__sentence"])
                    error_positions.append(pos)
                elif i["impression_sentence"]!="":
                    pos=data["report_text"][k].find(i["impression__sentence"])
                    error_positions.append(pos)
                else:
                    error_positions.append(-1)
        # print(error_prediction)
        final_data.at[k,"prediction"]=error_prediction
        final_data.at[k,"error_type_prediction"]=error_type_predictions
        final_data.at[k,"error_position_prediction"]=error_positions
        end = time.time()
        final_data.at[k,"total_time"]=end-start        

        # print("successful for "+ str(k))
        print("prediction for k "+ str(k) + " is "+ str(final_data["prediction"][k]))

    except:
        print("unsuccessful for "+ str(k))
        final_data.at[k,"report_index"]=report_index
        final_data.at[k,"report"]=input
        final_data.at[k,"prediction"]=0
        final_data.at[k,"errors"]=[]
        final_data.at[k,"error_type_prediction"]=[0]
        final_data.at[k,"error_position_prediction"]=[-1]   
        final_data.loc[k,"total_time"]=0     
        # print("failed for "+ str(k))
        print("prediction for k "+ str(k) + " is "+ str(final_data["prediction"][k]))


final_data["label"].fillna(0,inplace=True)
final_data["error_type_label"].fillna(0,inplace=True)
final_data["error_pos_label"].fillna(-1,inplace=True)
final_data

final_data["error_type_label_specific"]=final_data["error_type_label"]
final_data

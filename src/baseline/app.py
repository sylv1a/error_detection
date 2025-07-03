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
config_path="/home/srawat@FuL.med.uni-muenchen.de/project/llm-pathology-extraction/config/config.yaml"
llm = CustomLLM()
llm.load_config(config_path)
parser = JsonOutputParser()
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
def general(report_text):
    chain1_time=0
    chain2_time=0
    chain3_time=0
    chain4_time=0
    input1=report_text
    error_detection_prompt1="/home/srawat@FuL.med.uni-muenchen.de/project/llm-pathology-extraction/advanced_approach/prompts/table_creation.txt"
    error_detection_prompt="/home/srawat@FuL.med.uni-muenchen.de/project/llm-pathology-extraction/advanced_approach/prompts/table_creation2.txt"
    prompt1 = PromptTemplate(
    template="{prompt1}{input1}{prompt2}",
    input_variables=["input1"],
    partial_variables={
        "prompt1": load_prompt(error_detection_prompt1),
        "prompt2": load_prompt(error_detection_prompt)
    },
    )
    start1=time.time()
    chain1 = prompt1 | llm
    response1 = chain1.invoke(
        {"input1": input1}
        )
    end1=end1=time.time()
    import json
    response1_output=json.loads(response1.split("<output>")[1].split("</output>")[0])
    chain1_time=(end1-start1)
    #print("Table created. Total time taken = "+ str(end1-start1))
    
    response1_output1=[]
    response1_output2=[]
    for i in response1_output:
        if i["findings_sentence"]=="" or i["impression_sentence"]=="":
            response1_output2.append(i)
        else:
            response1_output1.append(i)

    if len(response1_output1)>0:
        #print("1. Pairs present")
        error_detection_prompt2="/home/srawat@FuL.med.uni-muenchen.de/project/llm-pathology-extraction/advanced_approach/prompts/plausibility_check.txt"
        input2=response1_output1
        prompt2 = PromptTemplate(
        template="{prompt1}{input1}{prompt2}",
        input_variables=["input1"],
        partial_variables={
            "prompt1": load_prompt(error_detection_prompt2),
            "prompt2": load_prompt(error_detection_prompt)
        },
        )
        start2=time.time()
        chain2 = prompt2 | llm
        response2 = chain2.invoke(
            {"input1": input2}
            )
        end2=time.time()
        chain2_time=end2-start2
        #print("Plausibility check done. Time taken = "+ str(end2-start2))
        response2_output=json.loads(response2.split("<output>")[1].split("</output>")[0])
        response2_output=[item for item in response2_output if item["plausibility_score"] <=20]
        for i in response2_output:
            if "condition_index" in i:
                del i["condition_index"]
            if "plausibility_score" in i:
                del i["plausibility_score"]

        if(len(response2_output)>0):
            #print("1.1 Implausible pairs present")
            errors1=response2_output
        else:
            #print("1.2 Plausible pairs")
            errors1=[]
        #print("Errors detected = "+str(len(errors1)))
    
    else:
        #print("1. No pairs present")
        errors1=[]
        #print("Errors detected = "+str(len(errors1)))
    
    if(len(response1_output2)>0):
        #print("2. Single sentences present")
        response1_output2_cleaned=[]
        for i in response1_output2:
            i={key: value for key, value in i.items() if value != ""}
            response1_output2_cleaned.append(i)    
        response1_output2_cleaned_findings=[]
        response1_output2_cleaned_impression=[]
        for i in response1_output2_cleaned:
            if("findings_sentence" in i.keys()):
                response1_output2_cleaned_findings.append(i)
            else:
                response1_output2_cleaned_impression.append(i)
        if (len(response1_output2_cleaned_findings)>0):
            #print("Single findings sentence present")
            input3=response1_output2_cleaned_findings
            impression=input1.split("\\")[1]
            error_detection_prompt4="/home/srawat@FuL.med.uni-muenchen.de/project/llm-pathology-extraction/advanced_approach/prompts/single_check_findings.txt"
            error_detection_prompt5="/home/srawat@FuL.med.uni-muenchen.de/project/llm-pathology-extraction/advanced_approach/prompts/single_check_findings2.txt"
            error_detection_prompt6="/home/srawat@FuL.med.uni-muenchen.de/project/llm-pathology-extraction/advanced_approach/prompts/single_check_findings3.txt"
            prompt3 = PromptTemplate(
                template="{prompt1}{input1}{prompt2}{input2}{prompt3}",
                input_variables=["input1","input2"],
                partial_variables={
                    "prompt1": load_prompt(error_detection_prompt4),
                    "prompt2": load_prompt(error_detection_prompt5),
                    "prompt3": load_prompt(error_detection_prompt6)
                },
            )
            chain3 = prompt3 | llm
            start3=time.time()
            response3 = chain3.invoke(
                {"input1": impression,
                "input2":input3} 
                )
            end3=time.time()
            chain3_time=end3-start3
            #print("Findings check done. Time taken = "+str(chain3_time))
            response3_output=json.loads(response3.split("<output>")[1].split("</output>")[0])
            response3_output=[item for item in response3_output if item["contradicts"].lower() =="yes"]
            errors2=response3_output
            #print("Errors detected = "+str(len(errors1+errors2)))
        else:
            #print("No single findings sentences present")
            errors2=[]
            #print("Errors detected = "+str(len(errors1+errors2)))
        if(len(response1_output2_cleaned_impression)>0):
            #print("Single impression sentences present")
            input4=response1_output2_cleaned_impression
            findings=input1.split("\\")[0]
            error_detection_prompt7="/home/srawat@FuL.med.uni-muenchen.de/project/llm-pathology-extraction/advanced_approach/prompts/single_check_impression.txt"
            error_detection_prompt8="/home/srawat@FuL.med.uni-muenchen.de/project/llm-pathology-extraction/advanced_approach/prompts/single_check_impression2.txt"
            error_detection_prompt9="/home/srawat@FuL.med.uni-muenchen.de/project/llm-pathology-extraction/advanced_approach/prompts/single_check_impression3.txt"
            prompt4 = PromptTemplate(
                template="{prompt1}{input1}{prompt2}{input2}{prompt3}",
                input_variables=["input1","input2"],
                partial_variables={
                    "prompt1": load_prompt(error_detection_prompt7),
                    "prompt2": load_prompt(error_detection_prompt8),
                    "prompt3": load_prompt(error_detection_prompt9)
                },
            )
            chain4 = prompt4 | llm
            start4=time.time()
            response4 = chain4.invoke(
                {"input1": findings,
                "input2":input4} 
                )
            end4=time.time()
            chain4_time=end4-start4
            #print("Impression check done. Total time taken = "+str(chain4_time))
            response4_output=json.loads(response4.split("<output>")[1].split("</output>")[0])
            response4_output=[item for item in response4_output if item["contradicts"].lower() =="yes"]
            errors3=response4_output
            #print("Errors detected = "+str(len(errors1+errors2+errors3)))
        else:
            #print("No single impression sentences present")
            errors3=[]
            #print("Errors detected = "+str(len(errors1+errors2+errors3)))
    else:
        #print("No single sentences present")
        errors2=[]
        errors3=[]
        #print("Errors detected = "+str(len(errors1+errors2+errors3)))
    errors=errors1+errors2+errors3
    total_time=chain1_time+chain2_time+chain3_time+chain4_time
    if(len(errors)>0):
        label=1
    else:
        label=0
    return(errors, total_time, label)

def others(report_text):
    start=time.time()
    input=report_text
    error_detection_prompt="/home/srawat@FuL.med.uni-muenchen.de/project/llm-pathology-extraction/advanced_approach/prompts/others.txt"
    error_detection_prompt2="/home/srawat@FuL.med.uni-muenchen.de/project/llm-pathology-extraction/advanced_approach/prompts/others2.txt"
    prompt = PromptTemplate(
        template="{prompt1}{input1}{prompt2}",
        input_variables=["input1"],
        partial_variables={
            "prompt1": load_prompt(error_detection_prompt),
            "prompt2": load_prompt(error_detection_prompt2)
        },
    )
    chain = prompt | llm
    response = chain.invoke(
        {"input1": input}
    )
    response_output=json.loads(response.split("<output>")[1].split("</output>")[0])
    response_output=[item for item in response_output if (item["error_label"]==1 and item["confidence_score"] >=80)]
    if(len(response_output)>0):
        label=1
        errors=response_output
    else:
        label=0
        errors=[]
    end=time.time()
    total_time=end-start
    return(errors, total_time, label)
def side_change(report_text):
    start=time.time()
    error_detection_prompt="/home/srawat@FuL.med.uni-muenchen.de/project/llm-pathology-extraction/advanced_approach/prompts/side_change_error_prompt.txt"
    error_detection_prompt2="/home/srawat@FuL.med.uni-muenchen.de/project/llm-pathology-extraction/advanced_approach/prompts/side_change_error_prompt2.txt"
    input=report_text
    prompt = PromptTemplate(
    template="{prompt1}{input1}{prompt2}",
    input_variables=["input1"],
    partial_variables={
        "prompt1": load_prompt(error_detection_prompt),
        "prompt2": load_prompt(error_detection_prompt2)
    },
    )
    chain = prompt | llm
    response = chain.invoke(
        {"input1": input}  # load_input(input_path) for .txt files
    )
    response_output=json.loads(response.split("<output>")[1].split("</output>")[0])
    response_output=[item for item in response_output if (item["error_label"]==1)]
    if len(response_output)>0:
        label=1
        errors=response_output
    else:
        label=0
        errors=[]
    end=time.time()
    total_time=end-start
    return(errors, total_time, label)


def app2(report):
    try:
        print("Date and measurement check in progress...")
        e1, t1, l1=others(report)
        error_type_predictions=[]
        error_positions=[]
        if l1==1:
            for k in e1:
                error_type_predictions.append(k["error_type_label"])     
                if k["findings_sentence"]!="" and k["impression_sentence"]!="":
                    pos=report.find(k["findings_sentence"])
                    error_positions.append(pos)
                    pos=report.find(k["impression_sentence"])
                    error_positions.append(pos)
                else:
                    if k["findings_sentence"]!="":
                        pos=report.find(k["findings_sentence"])
                        error_positions.append(pos)
                    else:
                        pos=report.find(k["impression_sentence"])
                        error_positions.append(pos)
        print("Date and measurement check done. Time taken= "+str(t1))
        print("Contradiction check in progress. This might take few minutes depending on cluster load...")
        e2, t2, l2=general(report)
        if l2==1:
            error_type_predictions.append(6)
            for k in e2:
                if (("findings_sentence" in k.keys()) and ("impression_sentence" not in k.keys())):
                    k["impression_sentence"]=""
                if (("findings_sentence" not in k.keys()) and ("impression_sentence" in k.keys())):
                    k["findings_sentence"]=""
            
            for k in e2:     
                if k["findings_sentence"]!="" and k["impression_sentence"]!="":
                    pos=report.find(k["findings_sentence"])
                    error_positions.append(pos)
                    pos=report.find(k["impression_sentence"])
                    error_positions.append(pos)
                else:
                    if k["findings_sentence"]!="":
                        pos=report.find(k["findings_sentence"])
                        error_positions.append(pos)
                    else:
                        pos=report.find(k["impression_sentence"])
                        error_positions.append(pos)
        print("Contradiction check done. Time taken= "+str(t2))
        
        total_errors=len(e1)+len(e2)
        if l1==1 or l2==1:
            prediction=1
        else:
            prediction=0
            error_type_predictions.append(0)
            error_positions.append(-1)

        error_pos=error_positions
        error_type=error_type_predictions
        errors=e1+e2
        total_time=t1+t2
        print("\n")
        print("Total errors detected = "+str(total_errors))
        for i in errors:
            print(i)
            print("\n")
    except Exception as e:
        print(e)
        print("Detection failed. Please try again.")

def app1(report):
    try:
        print("Date and measurement check in progress...")
        e1, t1, l1=others(report)
        error_type_predictions=[]
        error_positions=[]
        if l1==1:
            for k in e1:
                error_type_predictions.append(k["error_type_label"])     
                if k["findings_sentence"]!="" and k["impression_sentence"]!="":
                    pos=report.find(k["findings_sentence"])
                    error_positions.append(pos)
                    pos=report.find(k["impression_sentence"])
                    error_positions.append(pos)
                else:
                    if k["findings_sentence"]!="":
                        pos=report.find(k["findings_sentence"])
                        error_positions.append(pos)
                    else:
                        pos=report.find(k["impression_sentence"])
                        error_positions.append(pos)
        print("Date and measurement check done. Time taken= "+str(t1))
        print("Side change check in progress...")
        e2, t2, l2=side_change(report)
        if l2==1:
            for k in e2:
                error_type_predictions.append(3)     
                if k["findings_sentence"]!="" and k["impression_sentence"]!="":
                    pos=report.find(k["findings_sentence"])
                    error_positions.append(pos)
                    pos=report.find(k["impression_sentence"])
                    error_positions.append(pos)
                else:
                    if k["findings_sentence"]!="":
                        pos=report.find(k["findings_sentence"])
                        error_positions.append(pos)
                    else:
                        pos=report.find(k["impression_sentence"])
                        error_positions.append(pos)
        print("Side change check done. Time taken= "+str(t2))
        
        total_errors=len(e1)+len(e2)
        if l1==1 or l2==1:
            prediction=1
        else:
            prediction=0
            error_type_predictions.append(0)
            error_positions.append(-1)

        error_pos=error_positions
        error_type=error_type_predictions
        errors=e1+e2
        total_time=t1+t2
        print("\n")
        print("Total errors detected = "+str(total_errors))
        for i in errors:
            print(i)
            print("\n")
    except Exception as e:
        print(e)
        print("Detection failed. Please try again.")
print("\nSelect 1 for date, measurement and side change detection. And 2 for date, measurement and contradiction detection.")
print("Note - choice 2 covers much more scenarios but comes with tradeoffs - takes more time, might flag trivial errors at times, might not be successful at first try due to output parsing errors in between llms.")
print("For accuracy and limited scope use 1 otherwise use 2.")
choice=input("Input choice 1 or 2 - ")
print("\n")
if int(choice)==1:
    print("Please provide the report for error detection. Note that report should have a findings section followed by impression section, both separated by \\.") 
    print("Format - findings_text \\ impression_text")
    report=input("Provide report - ")    
    if report.find("\\")>0:
        if report.find("\\\\")>0:
            print("Wrong report format. Use single \\")
        else:
            app1(report)
    else:
        print("Wrong report format")
elif int(choice)==2:
    print("Please provide the report for error detection. Note that report should have a findings section followed by impression section, both separated by \\.") 
    print("Format - findings_text \\ impression_text")
    report=input("Provide report - ")    
    if report.find("\\")>0:
        if report.find("\\\\")>0:
            print("Wrong report format. Use single \\")
        else:
            app2(report)
    else:
        print("Wrong report format")
else:
    print("Invalid choice")
#'Multiple previous recordings, most recently from the day before. Increasingly flat consolidations on both sides basal, the Hili with unclearly demarcated recessus, consistent with draining pleural effusions in. Adjacent hypoventilations, especially accompanying beginning infiltrates. Pneumothorax discernible in lying position. No further significant finding changes. \ Increasingly draining pleural effusions on both sides with adjacent hypoventilations, especially accompanying beginning infiltrate.'
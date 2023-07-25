import openai
import os
import transformers
import json
import random
import argparse
import time
from tqdm import tqdm
import string
import sys
import csv
import jsonlines
import pdb
pdb.set_trace()
from datasets import load_dataset
openai.api_key="you api key"
demonstrations_set={
    'strategyqa':{
        'rephrase':"Rephrase the question and answer it.\n\n"+\
    "question:Are more people today related to Genghis Khan than Julius Caesar?\nrephrase:Do more people today have connection with Genghis Khan than Julius Caesar?\nanswer:True\n"+\
    "question:Would a dog respond to bell before Grey seal?\nrephrase: Would Grey seal respond to bell later than a dog?\nanswer:True\n"+\
    "question:Is a Boeing 737 cost covered by Wonder Woman (2017 film) box office receipts?\nrephrase:Does Wonder Woman box office receipts cover a Boeing 737 cost?\nanswer:True\n"+\
    "question:Is the language used in Saint Vincent and the Grenadines rooted in English?\nrephrase: Does the language used in Saint Vincent and the Grenadines originate from English?\nanswer:True\n"+\
    "question:Are Christmas trees dissimilar to deciduous trees?\nrephrase:Are Christmas trees different from deciduous trees?\nanswer:True\n"+\
    "question:Does Dragon Ball shows and movies fall short of Friday 13th number of projects?\nrephrase:Does Dragon Ball make less shows and movies than Friday 13th?\nanswer:True\n",

        'similar':"Given a context, generate a similar question to the given question and answer it.\n\n"+\
"context:A plum tree is a deciduous tree that bears fruit. Deciduous trees shed their leaves in the autumn. Autumn happens from September until the end of Deember.\nquestion:Is November a bad time for a photographer to take pictures of a plum tree in bloom?\nsimilar:Will the leaves a plum tree fall in the autumn?\nanswer:True\n"+\
"context:The Alamo is located in San Antonio. The Alamo was the site of a major battle during the Texan Revolution against Mexico in 1836.\nquestion:Was San Antonio the site of a major battle in the 19th century?\nsimilar:Was the Alamo the site of a major battle in the 19th century?\nanswer:True\n"+\
"context:Filicide is the act of killing a son or a daughter. Marvin Gay Sr. committed filicide in 1984 when he shot his son, singer Marvin Gaye. Isaac's father Abraham, was commanded by God to sacrifice his son Isaac, but was spared by an angel.\nquestion:Did Isaac's father almost commit similar crime as Marvin Gay Sr.?\ngsimilar:Did Isaac's father almost commit filicide?\nanswer:True\n"+\
"context:The animals that Yetis are said to look similar to are able to use their hands or toes to grasp items The ability to grasp with hands or other limbs is to be prehensile. \nquestion:Would a Yeti be likely to have prehensile limbs?\nsimilar:Will a Yeti fail to grasp items with its hands or toes?\nanswer:True\n"+\
"context:Land of Israel was controlled by the Ottoman Empire in 16th century.  The religion of Ottoman Empire was Sunni Islam. \nquestion:Was Land of Israel in possession of an Islamic empire in 16th century?\nsimilar:Was the Ottoman Empire Islamic once?\nanswer:True\n"+\
"context:Wedding rings are typically made of precious shiny stones such as diamonds. Silicon is a solid rock like element at room temperature that has a natural lustre. Bromine is a liquid at room temperature that is toxic to the touch.\nquestion:Will silicon wedding rings outsell bromine wedding rings?\nsimilar:Are silicon wedding rings shiny?\nanswer:True\n",

        'counterfactual':"Given the context and question, generate a question that negates the question.\n\n"+\
"context:A plum tree is a deciduous tree that bears fruit. Deciduous trees shed their leaves in the autumn. Autumn happens from September until the end of Deember.\nquestion:Is November a bad time for a photographer to take pictures of a plum tree in bloom?\ngenerate:Is a plum tree in bloom in the autumn?\n"+\
"context:The animals that Yetis are said to look similar to are able to use their hands or toes to grasp items The ability to grasp with hands or other limbs is to be prehensile. \nquestion:Would a Yeti be likely to have prehensile limbs?\ngenerate:Is a Yeti able to grasp items with its hands or toes?\n"+\
"context:Keelhauling was a severe punishment whereby the condemned man was dragged beneath the ship\u2019s keel on a rope. Keelhauling is considered a form of torture. Torture is considered cruel. The Eighth Amendment forbids the use of \"cruel and unusual punishment\\nquestion:Would keelhauling be a fair punishment under the Eighth Amendment?\ngenerate:Would keelhauling be considered cruel?\n"+\
"context:Khanbaliq was the winter capital of the Mongol Empire.  Khanbaliq was located at the center of what is now modern day Beijing, China. Moon Jae-In was born in Geoje, South Korea.\nquestion:Was Moon Jae-in born outside of Khanbaliq?\ngenerate:Was Moon Jae-in born in Beijing?\n"+\
"context:Amazonas is mostly tropical jungle. Tropical jungles contain dangerous creatures. Dangerous creatures put people's lives at risk.\nquestion:Does walking across Amazonas put a person's life at risk?\ngenerate:Is Amazonas a safe place?\n"+\
"context:The Los Angeles Memorial Sports Arena had a capacity of 16,740 people. Coachella has had attendance numbers in excess of 99.000 people. Coachella relies on an outdoor set up to accommodate the massive crowds.\nquestion:Was Los Angeles Memorial Sports Arena hypothetically inadequate for hosting Coachella?\ngenerate:Would Los Angeles Memorial Sports Arena be too big for Coachella?\n"

    },
    'openbookqa':{
        'rephrase':"Rephrase the given question:\n\n"+\
    "question:As a plant's roots get bigger, they split apart\nrephrase:As the roots of a plant grow larger, they tend to separate\n"+\
    "question:A magnet will stick to\nrephrase:To what will a magnet stick?\n"+\
    "question:Where are you likely to find a nonrenewable resource?\nrephrase: In what location are nonrenewable resources typically found?\n"+\
    "question:A man is searching for his dog in the woods and brings a flashlight. The flashlight has two large batteries in it, which\nrephrase:While searching for his missing dog in the woods, a man brings along a flashlight equipped with two large batteries, which\n"+\
    "question:Why do berries exist?\nrephrase:What is the purpose or reason for the existence of berries?\n"+\
    "question:which one of these can help a person cook their food?\nrephrase:Which of these options can assist someone in cooking their food?\n",

        'similar':"Generate a similar multiple choice question given the context and answer it.\n\n"+\
    "context:the sun is the source of energy for physical cycles on Earth.\nquestion:The sun is responsible for?\generate:Which of the following is the sun responsible for?(a)The water cycle(b)The carbon cycle(c)The nitrogen cycle(d)All of the above\nanswer:(d)\n"+\
    "context:as a source of light becomes closer , that source will appear brighter.\nquestion:As a car approaches you in the night\ngenerate:Which of the following statements is true as a car approaches you in the night?(a)The car's headlights will appear dimmer(b)The car's headlights will appear brighter(c)The car's headlights will appear the same brightness(d)It is impossible to determine the brightness of the car's headlights\nanswer:(b)\n"+\
    "context:natural gas is a nonrenewable resource\nquestion:Where are you likely to find a nonrenewable resource?\ngenerate:Which of the following statements is true about natural gas?(a)it is a renewable resource(b)It is an inexhaustible resource(c)It is a nonrenewable resource(d)It is a renewable resource that can be replenished quickly\nanswer:(c)\n"+\
    "context:a force acting on an object in the opposite direction that the object is moving can cause that object 's speed to decrease in a forward motion\nquestion: A car is driving on a highway trying to get up the speed limit. However, there are strong winds hitting the windshield of the car. Even though the driver is trying to speed up, the high winds?\n(a)The car will be able to reach the speed limit faster(b)The car's speed will remain unchanged(c)The car's speed will increase(d)The car's speed will decrease\nanswer:(d)\n"+\
    "context:An example of moisture is water vapor in the atmosphere\nquestion:Water vapor is an example of what?\ngenerate:Which of the following is an example of moisture?(a)A damp towel(b)A dry piece of paper(c)Water vapor in the atmosphere(d)Ice cubes in a drink\nanswer:(c)\n"+\
    "context:mammals give birth to live young\nquestion:Which animal gives birth to live young?\generate:(a)fish(b)reptiles(c)birds(d)tigers\nanswer:(d)\n"
    }
}
def clean_result(results,question_type):
    cleaned_results=[]
    multiple_results=results.split('\n\n')
    for result in multiple_results:
        sub_results=result.split('\n')
        if len(sub_results)>2:
            continue
        if question_type == 'similar':
            if len(sub_results)!=2:
                continue
            if sub_results[1][:6]!='answer':
                continue
        # if question_type in ['rephrase']:
        #     if len(sub_results)!=1:
        #         continue
        if 'context' in result:
            continue
        cleaned_results.append(result)
    return cleaned_results[:-2]


def generate_questions(inst,question_type,dataset,gpt3_version,max_tokens):
    result=''
    if dataset=='strategyqa':
        question=inst['question']
        context=' '.join(inst['facts'])
        # answer="true" if inst["answer"]==True else "false"
        if question_type=='rephrase':
            prompt="question:"+question+"\nrephrase:"
        elif question_type=='similar':
            prompt="context:"+context+"\nquestion:"+question+"\nsimilar:"
        elif question_type=='counterfactual':
            prompt="context:"+context+"\nquestion:"+question+"\ngenerate:"  
    
    elif dataset=='openbookqa':
        question=inst['question']['stem']
        context=inst['fact1']
        if question_type=='rephrase':
            prompt="question:"+question+"\nrephrase:"
        elif question_type=='similar':
            prompt="context:"+context+"\nquestion:"+question+"\ngenerate:"
        else:
            raise NotImplementedError
    response = openai.Completion.create(engine=gpt3_version, prompt=demonstrations_set[dataset][question_type]+prompt, max_tokens=max_tokens, temperature=0.7,n=5)
    res=response['choices']

    for one_res in res:
        result+=one_res['text'].split('\n\n')[0]+'\n\n'
    return {
            'context':context,
            'ori_q':question,
            'result':result        
        }   
       
    
def main(args):
    results=[]
    csvfile=open(args.output_file,'w',newline='')
    csvwriter=csv.writer(csvfile,delimiter=',')
    if args.question_type =='similar':
        csvwriter.writerow(['context','original question','generated question','answer'])
    else:
        csvwriter.writerow(['context','original question','generated question'])


    if args.dataset=='strategyqa':
        data_dir=os.path.join(args.data_dir,args.dataset,'raw',f'strategyqa_processed_{args.split}.json')
        data = json.load(open(data_dir,'r'))
        for inst in tqdm(data):
            generated_result=generate_questions(inst,args.question_type,args.dataset,args.gpt3_version,args.max_output_tokens)
            cleaned_result=clean_result(generated_result['result'],args.question_type)
            for result in cleaned_result:
                if args.question_type=='similar':
                    answer=result.split('\n')[1][7:]
                    csvwriter.writerow([generated_result['context'],generated_result['ori_q'],result.split('\n')[0],answer])
                else:
                    csvwriter.writerow([generated_result['context'],generated_result['ori_q'],result.split('\n')[0]])
    elif args.dataset=='openbookqa':
        data_dir=os.path.join(args.data_dir,args.dataset,'raw',f'{args.split}_complete.jsonl')
        with jsonlines.open(data_dir,'r') as f:

            for inst in tqdm(f.iter()):
                generated_result=generate_questions(inst,args.question_type,args.dataset,args.gpt3_version,args.max_output_tokens)
                cleaned_result=clean_result(generated_result['result'],args.question_type)
                for result in cleaned_result:
                    if args.question_type=='similar':
                        answer=result.split('\n')[1][7:]
                        csvwriter.writerow([generated_result['context'],generated_result['ori_q'],result.split('\n')[0],answer])
                    else:
                        csvwriter.writerow([generated_result['context'],generated_result['ori_q'],result])

    else:
        raise NotImplementedError

    

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessing.')
    parser.add_argument('--data_dir', default='../../data/',type=str)
    parser.add_argument('--split', type=str,choices=['train','test','dev'])
    parser.add_argument('--dataset', type=str,default='strategyqa',choices=['strategyqa','openbookqa'])
    parser.add_argument('--question_type',type=str,choices=['rephrase','similar','counterfactual'])
    parser.add_argument('--output_file', type=str) #defaut file format is csv
    parser.add_argument('--gpt3_version', type=str, default='davinci-instruct-beta')
    parser.add_argument('--max_output_tokens',type=int,default=100)
   
    
    args = parser.parse_args()
    main(args)
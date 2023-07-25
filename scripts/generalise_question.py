import openai
import os
import json
import argparse
import pandas as pd


def load_data(file):
    json_results=json.load(open(file,'r'))
    print('the numbers of instances is, ',len(json_results))
    prompts=[]
    for instance in json_results:
        input=instance['question']
        answer=instance['answer']
        facts=' '.join(instance['facts'])
        if answer==False:
            output='false. explanation: '+facts
        else:
            output='true. explanation: '+facts
        prompts.append([input,output])

    return prompts

def template_1(rationale, question):
    demonstration = "Paraphase the question given the context. \n\n context: When Gauss died in 1855, his brain was preserved for study. Dr. Rudolf Wagner, who studied the brain, found the mass to be slightly above average, and found highly developed convolutions on the brain. question: Did Gauss have a normal brain structure? paraphase: Was brain structure of Gauss normal? \n\n" + \
            "context: When milk becomes acidic, the water and fats separate from each other. When the water and fats separate in milk, it becomes clumpy and has a bad texture.Lemon is highly acidic. question: Does Lemon enhance the flavor of milk? paraphase: Will the flavor of milk be enhanced by lemon? \n\n" + \
            "context: People with heart failure have to limit their sodium intake. Ramen is notorious for having incredibly high sodium levels. question: Would ramen be bad for someone with heart failure? paraphase: Would ramen harm someone with heart failure? \n\n"
    
    prompt = f'{demonstration}context: {rationale} question: {question}. paraphrase:'

    return prompt


def main(args):
    # openai.api_key=os.getenv('OPENAI_API_KEY')
    openai.api_key='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'
    gpt3_version = args.gpt3_version

    if args.dataset == "strategyqa":
        path = os.path.join(args.data_dir, args.dataset, "raw", "strategyqa_processed_test.json")
        dataset = pd.read_json(path, orient='records')

        gold_rationales = []
        gold_questions = []
        paraphrases = []

        for index, row in dataset.iterrows():

            rationale = row['rationale']
            question = row['question']

            if args.template == 1:
                prompt = template_1(rationale, question)
                response = openai.Completion.create(engine=gpt3_version, prompt=prompt, max_tokens=100, temperature=0.0)
                response_text = response['choices'][0]['text']

            print(response_text)
            print()
            print(prompt)
            print("-------=-=-=-=-=-=-=----------")
            
            gold_rationales.append(rationale)
            gold_questions.append(question)
            paraphrases.append(response_text)
        
        save_path = os.path.join(args.data_dir, args.dataset, "raw", f'generalise_{str(args.template)}.csv')
        save_df = pd.DataFrame(data={"rationale":gold_rationales, "question":gold_questions, "paraphrase":paraphrases})
        save_df.to_csv(save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessing.')
    parser.add_argument('--data_dir', type=str, default='../data/', help='Root directory for datasets.')
    parser.add_argument('--dataset', type=str, choices=['strategyqa','openbookqa'])
    parser.add_argument('--template', type=int, choices=[1])
    # parser.add_argument('--test_file', type=str)
    # parser.add_argument('--train_file', type=str)
    # parser.add_argument('--output_file', type=str)
    parser.add_argument('--gpt3_version', type=str, default='davinci-instruct-beta')
    # parser.add_argument('--max_output_tokens',type=int,default=75)
    # parser.add_argument('--max_seq_len', type=int, default=1024)
    
    args = parser.parse_args()
    main(args)
import random
from src.utils.gpt3_utils import input_label_mapper


def feb_gpt3_prompt_creator(question, rationale, answer, choices, dataset):
    if dataset == "strategyqa":
        choices = ', '.join(choices)
    if dataset == "qed":
        input = f"Question: {question}\n" \
                "Answer:"
    else:
        input = f"Question: {question}\n" \
        f"Choices: {choices}\n" \
        "Answer:"
    if rationale != "":
        output = f" {input_label_mapper(answer, dataset)}\n" \
                f"Reason: {rationale}\n" \
                "###\n"
    else:
        output = f"{input_label_mapper(answer, dataset)}\n" \
                    "###\n"
    return input, output

def cot_gpt3_prompt_creator(question, rationale, answer, choices, dataset):
    if dataset == "strategyqa":
        choices = " or ".join(choices)
        header = f"Q: {choices}: "
        input = f"{header} {question}\nA:"
        if rationale:
            output = f"{rationale} The answer is {input_label_mapper(answer, dataset)}.\n\n"
        else:
            output = f"The answer is {input_label_mapper(answer, dataset)}.\n\n"
        return input, output
    elif dataset == "openbookqa":
        question_header = f"Q: {question}\n"
        answer_choices = f"Answer Choices: {choices}\n"
        input = f"{question_header} {answer_choices}\nA:"
        if rationale:
            output = f"{rationale}. So the answer is {input_label_mapper(answer, dataset)}.\n\n"
        else:
            output = f"The answer is {input_label_mapper(answer, dataset)}.\n\n"
        return input, output
    elif dataset == "qed":
        question_header = f"Q: {question}"
        input = f"{question_header}\nA:"
        if rationale:
            output = f"{rationale}So the answer is {input_label_mapper(answer, dataset)}.\n\n"
        else:
            output = f"The answer is {input_label_mapper(answer, dataset)}.\n\n"
        return input, output
        

def get_strategyqa_fixed_demonstration_set():
    questions = ["Do hamsters provide food for any animals?", \
                "Could Brooke Shields succeed at University of Pennsylvania?", \
                "Hydrogen's atomic number squared exceeds number of Spice Girls?", \
                "Is it common to see frost during some college commencements?",\
                "Could a llama birth twice during War in Vietnam (1945-46)?", \
                "Would a pear sink in water?"]
    rationales = ["Hamsters are prey animals. Prey animals provide food for predators.", \
                "Brooke Shields graduated from Princeton University. Princeton is ranked as the number 1 national college by US news. University of Pennsylvania is ranked as number 6 national college by US news. Princeton only admits around 6 percent of applicants as of 2018. University of Pennsylvania accepts around 9% of applicants as of 2018.", \
                "Hydrogen is the first element and has an atomic number of one. To square a number, you multiply it by itself. The Spice Girls has five members.", \
                "College commencement ceremonies often happen during the months of December, May, and sometimes June. Frost isn't uncommon to see during the month of December, as it is the winter.", \
                "The War in Vietnam (1945-46) lasted around 6 months. The gestation period for a llama is 11 months.", \
                "The density of a raw pear is about 0.59 g\/cm^3. The density of water is about 1 g\/cm^3. Objects only sink if they are denser than the surrounding fluid."]
    labels = [True, True, False, True, False, False]

    return questions, rationales, labels


def process_strategyqa_prompt(question, rationale, answer, gen_mode, incontext, prompt_type, train_prompts = None, max_prompt_length = None, tokenizer = None):
    if gen_mode == "I-O":
        if incontext == "cot":
            if prompt_type == "cot":
                demonstration = "Q: Do hamsters provide food for any animals?\nA: The answer is yes.\n\nQ: Could Brooke Shields succeed at University of Pennsylvania?\nA: The answer is yes.\n\nQ: Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?\nA: The answer is no.\n\nQ: Yes or no: Is it common to see frost during some college commencements?\nA: The answer is yes.\n\nQ: Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?\nA: The answer is no.\n\nQ: Yes or no: Would a pear sink in water?\nA: The answer is no.\n\nQ: Yes or no: "
                prompt = demonstration + question + "\nA:"
                label = input_label_mapper(answer)
                return prompt, label
            elif prompt_type == "feb":
                questions, _, labels = get_strategyqa_fixed_demonstration_set()

                demonstration = ''
                for i in range(len(questions)):
                    input, output = feb_gpt3_prompt_creator(questions[i],'',labels[i], ['Yes', 'No'], 'strategyqa')
                    demonstration += input + output
                
                prompt, label = feb_gpt3_prompt_creator(question, '',answer, ['Yes', 'No'], 'strategyqa')
                prompt = f'{demonstration}{prompt}'

                return prompt, label

        elif incontext == "feb_random":
            if prompt_type=='feb':
                assert train_prompts!=None
                assert max_prompt_length!=None
                task_description = "Answer the question from the provided choices,.\n"

                random.shuffle(train_prompts)
                header = task_description

                prompt, label = feb_gpt3_prompt_creator(question, '', answer, ['Yes', 'No'], 'strategyqa')

                current_seqlen = len(tokenizer.tokenize(f'{header}{prompt}')) + max_prompt_length

                for i,(train_instance_input,train_instance_output) in enumerate(train_prompts):
                    train_instance_seqlen = len(tokenizer.tokenize(f'{train_instance_input}{train_instance_output}'))
                    if current_seqlen + train_instance_seqlen > 2049:
                        break
                    header = header + train_instance_input + train_instance_output
                    current_seqlen += train_instance_seqlen

                prompt = f'{header}{prompt}'
                
                return prompt, label

        elif incontext == 'feb_6':
            if prompt_type=='feb':
                assert train_prompts!=None
                assert max_prompt_length!=None
                task_description = "Answer the question from the provided choices.\n"

                random.shuffle(train_prompts)
                header = task_description

                prompt, label = feb_gpt3_prompt_creator(question, '', answer, ['Yes', 'No'], 'strategyqa')

                for i,(train_instance_input,train_instance_output) in enumerate(train_prompts):
                    if i >= 6:
                        break
                    header = header + train_instance_input + train_instance_output

                prompt = f'{header}{prompt}'
                
                return prompt, label
        elif incontext == 'None':
            if prompt_type == 'infilling':
                prompt = f"answer strategyqa question: {question}" + " choice: ".join(['True','False']) 
                answer = f"<extra_id_0> {answer} <extra_id_1>"
                return prompt, answer
            elif prompt_type == 'squadt5':
                prompt = f"answer strategyqa question: {question} context: True, False "
                answer = f"{answer}"
                return prompt, answer
            elif prompt_type == 't5like':
                prompt = f"answer strategyqa query: {question} entities: True, False "
                answer= f"{answer}"
                return prompt, answer
            elif prompt_type == 'qasimple':
                choice_ids = ['(A)','(B)']
                prompt = f'answer {question.lower()} \\n'
                choices = ['True','False']
                for choice_id, choice in zip(choice_ids,choices):
                    prompt += f' {choice_id} {choice.lower()}'
                answer = f"{answer.lower()}"
                return prompt, answer
    elif gen_mode == "IR-O":
        if incontext == 'None':
            if prompt_type == "squadt5":
                 prompt = f"answer strategyqa question given the explanation: {question} context: True, False explanation: {rationale}"
                 answer = f"{answer}"
                 return prompt, answer
            elif prompt_type == 'infilling':
                prompt = f"answer strategyqa question given the explanation: {question} choices: " + " choice: ".join(['True','False']) + f" <extra_id_0> because <extra_id_1> {rationale} <extra_id_2>"
                answer = f"<extra_id_0> {answer} <extra_id_1>"
                return prompt, answer
                #feb 
                # input_string = f"explain {datasource} question: {question} choice: " + " choice: ".join(item["choices"]) + f" <extra_id_0> {explanation_sep} <extra_id_1>"
                # answer_string = f"<extra_id_0> {answer} <extra_id_1> {abstr_expl} <extra_id_2>"
            elif prompt_type == 't5like':
                prompt = f"answer strategyqa query given the explanation: {question} entities: True, False explanation: {rationale} "
                answer= f"{answer}"
                return prompt, answer
                 #feb
                # input_string = f"explain {datasource} query: {question} entities: " + ', '.join(item['choices']) # explain cos_e query: When getting in shape you need to have this in between workouts? entities: give up, period of recovery, jogging
                # answer_string = f"{answer} {explanation_sep} {abstr_expl}"

            elif prompt_type == 'qasimple':
                choice_ids = ['(A)','(B)']
                prompt = f'answer {question.lower()} given the explanation explanation: {rationale.lower()}'
                choices = ['True','False']
                for choice_id, choice in zip(choice_ids,choices):
                    prompt += f' {choice_id} {choice.lower()}'
                answer = f"{answer.lower()}"
                return prompt, answer
    elif gen_mode == "I-RO":
        if incontext == "cot":
            if prompt_type == "cot":
                demonstration = "Q: Do hamsters provide food for any animals?\nA: Hamsters are prey animals. Prey animals provide food for predators. The answer is yes.\n\n \
                                Q: Could Brooke Shields succeed at University of Pennsylvania?\nA: Brooke Shields graduated from Princeton University. Princeton is ranked as the number 1 national college by US news. University of Pennsylvania is ranked as number 6 national college by US news. Princeton only admits around 6 percent of applicants as of 2018. University of Pennsylvania accepts around 9% of applicants as of 2018. The answer is yes.\n\n\
                                Q: Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?\nA: Hydrogen is the first element and has an atomic number of one. To square a number, you multiply it by itself. The Spice Girls has five members. The answer is no.\n\n\
                                Q: Yes or no: Is it common to see frost during some college commencements?\nA: College commencement ceremonies often happen during the months of December, May, and sometimes June.  Frost isn't uncommon to see during the month of December, as it is the winter. The answer is yes.\n\n\
                                Q: Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?\nA: The War in Vietnam (1945-46) lasted around 6 months. The gestation period for a llama is 11 months. The answer is no.\n\n\
                                Q: Yes or no: Would a pear sink in water?\nA: The density of a raw pear is about 0.59 g\/cm^3. The density of water is about 1 g\/cm^3. Objects only sink if they are denser than the surrounding fluid. The answer is no.\n\n\
                                Q: Yes or no: "                
                prompt = demonstration + question + "\nA: "
                label = input_label_mapper(answer)
                rationale = rationale.strip()
                if rationale[-1] == ".":
                    label = " " + rationale + "The answer is " + label + "."
                else:
                    label = " " + rationale + ". The answer is " + label + "."
                return prompt, label

        elif incontext == "feb_6":
            if prompt_type == "cot":

                assert train_prompts!=None
                assert max_prompt_length!=None

                task_description = "Answer the question from the provided choices, and provide a reason why the answer is correct.\n"

                random.shuffle(train_prompts)
                header = task_description

                prompt, label = cot_gpt3_prompt_creator(question, rationale, answer, ['Yes', 'No'], 'strategyqa')

                for i,(train_instance_input,train_instance_output) in enumerate(train_prompts):
                    if i >= 6:
                        break
                    header = header + train_instance_input + train_instance_output

                prompt = f'{header}{prompt}'
                
                return prompt, label

        elif incontext == "feb_random":
            if prompt_type == "cot":

                assert train_prompts!=None
                assert max_prompt_length!=None

                task_description = "Answer the question from the provided choices, and provide a reason why the answer is correct.\n"

                random.shuffle(train_prompts)
                header = task_description

                prompt, label = cot_gpt3_prompt_creator(question, rationale, answer, ['Yes', 'No'], 'strategyqa')

                current_seqlen = len(tokenizer.tokenize(f'{header}{prompt}')) + max_prompt_length

                for i,(train_instance_input,train_instance_output) in enumerate(train_prompts):
                    train_instance_seqlen = len(tokenizer.tokenize(f'{train_instance_input}{train_instance_output}'))
                    if current_seqlen + train_instance_seqlen > 2049:
                        break
                    header = header + train_instance_input + train_instance_output
                    current_seqlen += train_instance_seqlen

                prompt = f'{header}{prompt}'
                
                return prompt, label

    elif gen_mode == "I-OR":
        if incontext == 'None':
            if prompt_type == "squadt5":
                 prompt = f"explain strategyqa question: {question} context: True, False "
                 answer = f"{answer} because {rationale}"
                 return prompt, answer
            elif prompt_type == 'infilling':
                prompt = f"explain strategyqa question: {question}" + " choice: ".join(['True','False']) + f" <extra_id_0> because <extra_id_1>"
                answer = f"<extra_id_0> {answer} <extra_id_1> {rationale} <extra_id_2>"
                return prompt, answer
                #feb 
                # input_string = f"explain {datasource} question: {question} choice: " + " choice: ".join(item["choices"]) + f" <extra_id_0> {explanation_sep} <extra_id_1>"
                # answer_string = f"<extra_id_0> {answer} <extra_id_1> {abstr_expl} <extra_id_2>"
            elif prompt_type == 't5like':
                prompt = f"explain strategyqa query: {question} entities: True, False "
                answer= f"{answer} because {rationale}"
                return prompt, answer
                 #feb
                # input_string = f"explain {datasource} query: {question} entities: " + ', '.join(item['choices']) # explain cos_e query: When getting in shape you need to have this in between workouts? entities: give up, period of recovery, jogging
                # answer_string = f"{answer} {explanation_sep} {abstr_expl}"

            elif prompt_type == 'qasimple':
                choice_ids = ['(A)','(B)']
                prompt = f'explain {question.lower()} \\n'
                choices = ['True','False']
                for choice_id, choice in zip(choice_ids,choices):
                    prompt += f' {choice_id} {choice.lower()}'
                answer = f"{answer.lower()} because {rationale.lower()}"
                return prompt, answer
                # choice_ids = ['(A)', '(B)', '(C)', '(D)', '(E)']
                # input_string = f'explain {question.lower()} \\n'
                # for choice_id, choice in zip(choice_ids, item["choices"]):
                #     input_string += f' {choice_id} {choice.lower()}'
                # answer_string = f"{answer.lower()} {explanation_sep} {abstr_expl.lower()}"
                # answer_string = answer_string.lower()

        elif incontext == "feb_random":
            if prompt_type == "feb":
                assert train_prompts!=None
                assert max_prompt_length!=None

                task_description = "Answer the question from the provided choices, and provide a reason why the answer is correct.\n"

                random.shuffle(train_prompts)
                header = task_description

                prompt, label = feb_gpt3_prompt_creator(question, rationale, answer, ['Yes', 'No'], 'strategyqa')

                current_seqlen = len(tokenizer.tokenize(f'{header}{prompt}')) + max_prompt_length

                for i,(train_instance_input,train_instance_output) in enumerate(train_prompts):
                    train_instance_seqlen = len(tokenizer.tokenize(f'{train_instance_input}{train_instance_output}'))
                    if current_seqlen + train_instance_seqlen > 2049:
                        break
                    header = header + train_instance_input + train_instance_output
                    current_seqlen += train_instance_seqlen

                prompt = f'{header}{prompt}'
                
                return prompt, label

        elif incontext == "feb_6":
            if prompt_type == "feb":

                assert train_prompts!=None
                assert max_prompt_length!=None

                task_description = "Answer the question from the provided choices, and provide a reason why the answer is correct.\n"

                random.shuffle(train_prompts)
                header = task_description

                prompt, label = feb_gpt3_prompt_creator(question, rationale, answer, ['Yes', 'No'], 'strategyqa')

                for i,(train_instance_input,train_instance_output) in enumerate(train_prompts):
                    if i >= 6:
                        break
                    header = header + train_instance_input + train_instance_output

                prompt = f'{header}{prompt}'
                
                return prompt, label

        elif incontext == "cot":
            if prompt_type == "feb":
                questions, rationales, labels = get_strategyqa_fixed_demonstration_set()
                demonstration = ""
                for i in range(len(questions)):
                    input, output = feb_gpt3_prompt_creator(questions[i], rationales[i], labels[i], ['Yes', 'No'], 'strategyqa')
                    demonstration += input + output
                
                prompt, label = feb_gpt3_prompt_creator(question, rationale, answer, ['Yes', 'No'], 'strategyqa')
                prompt = f'{demonstration}{prompt}'

                return prompt, label
    else:
        raise NotImplementedError

def process_openbookqa_choices(choices):
    label_list = choices['label']
    text_list = choices['text']
    choices_string = ""
    for l, t in zip(label_list,text_list):
        choices_string += f' ({l}) {t}'
    
    return choices_string

def process_openbookqa_prompt(question, choices, rationale, answer, gen_mode, incontext, prompt_type, train_prompts = None, max_prompt_length = None, tokenizer = None):
    if gen_mode == "I-O":
        if incontext == 'None':
            if prompt_type == 'infilling':
                prompt = f"answer openbookqa question: {question} choice: {choices}"
                answer = f"<extra_id_0> {answer} <extra_id_1>"
                return prompt, answer
            elif prompt_type == 'squadt5':
                prompt = f"answer openbookqa question: {question} context: {choices} "
                answer = f"{answer}"
                return prompt, answer
            elif prompt_type == 't5like':
                prompt = f"answer openbookqa query: {question} entities: {choices} "
                answer= f"{answer}"
                return prompt, answer
            elif prompt_type == 'qasimple':
                prompt = f'answer {question.lower()} {choices}\\n'
                answer = f"{answer}"
                return prompt, answer
    elif gen_mode == "IR-O":
        if incontext =="None":
            if prompt_type=="infilling":
                prompt = f"answer strategyqa question given the explanation: {question} choices: {choices} <extra_id_0> because <extra_id_1> {rationale} <extra_id_2>"
                answer = f"<extra_id_0> {answer} <extra_id_1>"
                return prompt, answer

    elif gen_mode == "I-OR":
        if incontext == 'None':
            if prompt_type == "squadt5":
                prompt = f"explain openbookqa question: {question} context: {choices}"
                answer = f"{answer} because {rationale}"
                return prompt, answer
            elif prompt_type == 'infilling':
                prompt = f"explain openbookqa question: {question} choice: {choices} <extra_id_0> because <extra_id_1>"
                answer = f"<extra_id_0> {answer} <extra_id_1> {rationale} <extra_id_2>"
                return prompt, answer
            elif prompt_type == 't5like':
                prompt = f"explain openbookqa query: {question} entities: {choices} "
                answer= f"{answer} because {rationale}"
                return prompt, answer
            elif prompt_type == 'qasimple':
                prompt = f'explain {question.lower()} {choices}\\n'
                answer = f"{answer} because {rationale.lower()}"
                return prompt, answer

        elif incontext == "feb_random":
            if prompt_type == "feb":
                assert train_prompts!=None
                assert max_prompt_length!=None

                task_description = "Answer the question from the provided choices, and provide a reason why the answer is correct.\n"

                random.shuffle(train_prompts)
                header = task_description

                prompt, label = feb_gpt3_prompt_creator(question, rationale, answer, choices, 'openbookqa')

                current_seqlen = len(tokenizer.tokenize(f'{header}{prompt}')) + max_prompt_length

                for i,(train_instance_input,train_instance_output) in enumerate(train_prompts):
                    train_instance_seqlen = len(tokenizer.tokenize(f'{train_instance_input}{train_instance_output}'))
                    if current_seqlen + train_instance_seqlen > 2049:
                        break
                    header = header + train_instance_input + train_instance_output
                    current_seqlen += train_instance_seqlen

                prompt = f'{header}{prompt}'
                
                return prompt, label

        elif incontext == "feb_6":
            if prompt_type == "feb":

                assert train_prompts!=None
                assert max_prompt_length!=None

                task_description = "Answer the question from the provided choices, and provide a reason why the answer is correct.\n"

                random.shuffle(train_prompts)
                header = task_description

                prompt, label = feb_gpt3_prompt_creator(question, rationale, answer, choices, 'openbookqa')

                for i,(train_instance_input,train_instance_output) in enumerate(train_prompts):
                    if i >= 6:
                        break
                    header = header + train_instance_input + train_instance_output

                prompt = f'{header}{prompt}'
                
                return prompt, label

    elif gen_mode == "I-RO":
        if incontext == "feb_6":
            if prompt_type == "cot":

                assert train_prompts!=None
                assert max_prompt_length!=None

                task_description = "Answer the question from the provided choices, and provide a reason why the answer is correct.\n"

                random.shuffle(train_prompts)
                header = task_description

                prompt, label = cot_gpt3_prompt_creator(question, rationale, answer, choices, 'openbookqa')

                for i,(train_instance_input,train_instance_output) in enumerate(train_prompts):
                    if i >= 6:
                        break
                    header = header + train_instance_input + train_instance_output

                prompt = f'{header}{prompt}'
                
                return prompt, label

        elif incontext == "feb_random":
            if prompt_type == "cot":

                assert train_prompts!=None
                assert max_prompt_length!=None

                task_description = "Answer the question from the provided choices, and provide a reason why the answer is correct.\n"

                random.shuffle(train_prompts)
                header = task_description

                prompt, label = cot_gpt3_prompt_creator(question, rationale, answer, choices, 'openbookqa')

                current_seqlen = len(tokenizer.tokenize(f'{header}{prompt}')) + max_prompt_length

                for i,(train_instance_input,train_instance_output) in enumerate(train_prompts):
                    train_instance_seqlen = len(tokenizer.tokenize(f'{train_instance_input}{train_instance_output}'))
                    if current_seqlen + train_instance_seqlen > 2049:
                        break
                    header = header + train_instance_input + train_instance_output
                    current_seqlen += train_instance_seqlen

                prompt = f'{header}{prompt}'
                
                return prompt, label

    else:
        raise NotImplementedError

def process_qed_prompt(question, rationale, answer, gen_mode, incontext, prompt_type, train_prompts = None, max_prompt_length = None, tokenizer = None):
    if gen_mode == "I-O":
        if incontext == 'None':
            if prompt_type == 'infilling':
                prompt = f"answer qed question: {question}"
                answer = f"<extra_id_0> {answer} <extra_id_1>"
                return prompt, answer
            elif prompt_type == 'squadt5':
                prompt = f"answer qed question: {question} "
                answer = f"{answer}"
                return prompt, answer
            elif prompt_type == 't5like':
                prompt = f"answer qed query: {question} "
                answer= f"{answer}"
                return prompt, answer
            elif prompt_type == 'qasimple':
                prompt = f'answer {question.lower()}\\n'
                answer = f"{answer}"
                return prompt, answer
            
    elif gen_mode == "I-OR":
        if incontext == 'None':
            if prompt_type == "squadt5":
                prompt = f"explain qed question: {question} "
                answer = f"{answer} because {rationale}"
                return prompt, answer
            elif prompt_type == 'infilling':
                prompt = f"explain qed question: {question} <extra_id_0> because <extra_id_1>"
                answer = f"<extra_id_0> {answer} <extra_id_1> {rationale} <extra_id_2>"
                return prompt, answer
            elif prompt_type == 't5like':
                prompt = f"explain qed query: {question} "
                answer= f"{answer} because {rationale}"
                return prompt, answer
            elif prompt_type == 'qasimple':
                prompt = f'explain {question.lower()}\\n'
                answer = f"{answer} because {rationale.lower()}"
                return prompt, answer

        elif incontext == "feb_random":
            if prompt_type == "feb":
                assert train_prompts!=None
                assert max_prompt_length!=None

                task_description = "Answer the question from the provided choices, and provide a reason why the answer is correct.\n"

                random.shuffle(train_prompts)
                header = task_description

                prompt, label = feb_gpt3_prompt_creator(question, rationale, answer, None, 'qed')

                current_seqlen = len(tokenizer.tokenize(f'{header}{prompt}')) + max_prompt_length

                for i,(train_instance_input,train_instance_output) in enumerate(train_prompts):
                    train_instance_seqlen = len(tokenizer.tokenize(f'{train_instance_input}{train_instance_output}'))
                    if current_seqlen + train_instance_seqlen > 2049:
                        break
                    header = header + train_instance_input + train_instance_output
                    current_seqlen += train_instance_seqlen

                prompt = f'{header}{prompt}'
                
                return prompt, label

        elif incontext == "feb_6":
            if prompt_type == "feb":

                assert train_prompts!=None
                assert max_prompt_length!=None

                task_description = "Answer the question from the provided choices, and provide a reason why the answer is correct.\n"

                random.shuffle(train_prompts)
                header = task_description

                prompt, label = feb_gpt3_prompt_creator(question, rationale, answer, None, 'qed')

                for i,(train_instance_input,train_instance_output) in enumerate(train_prompts):
                    if i >= 6:
                        break
                    header = header + train_instance_input + train_instance_output

                prompt = f'{header}{prompt}'
                
                return prompt, label

    elif gen_mode == "I-RO":
        if incontext == "feb_6":
            if prompt_type == "cot":

                assert train_prompts!=None
                assert max_prompt_length!=None

                task_description = "Answer the question from the provided choices, and provide a reason why the answer is correct.\n"

                random.shuffle(train_prompts)
                header = task_description

                prompt, label = cot_gpt3_prompt_creator(question, rationale, answer, None, 'qed')

                for i,(train_instance_input,train_instance_output) in enumerate(train_prompts):
                    if i >= 6:
                        break
                    header = header + train_instance_input + train_instance_output

                prompt = f'{header}{prompt}'
                
                return prompt, label

        elif incontext == "feb_random":
            if prompt_type == "cot":

                assert train_prompts!=None
                assert max_prompt_length!=None

                task_description = "Answer the question from the provided choices, and provide a reason why the answer is correct.\n"

                random.shuffle(train_prompts)
                header = task_description

                prompt, label = cot_gpt3_prompt_creator(question, rationale, answer, None, 'qed')

                current_seqlen = len(tokenizer.tokenize(f'{header}{prompt}')) + max_prompt_length

                for i,(train_instance_input,train_instance_output) in enumerate(train_prompts):
                    train_instance_seqlen = len(tokenizer.tokenize(f'{train_instance_input}{train_instance_output}'))
                    if current_seqlen + train_instance_seqlen > 2049:
                        break
                    header = header + train_instance_input + train_instance_output
                    current_seqlen += train_instance_seqlen

                prompt = f'{header}{prompt}'
                
                return prompt, label

    else:
        raise NotImplementedError
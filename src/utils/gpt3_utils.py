import string

def get_API_key():
    return "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"

def _parse_label_text(text, dataset):
    splitted_string = text.strip().split(" ")[-1]
    return output_label_mapper(splitted_string.translate(str.maketrans('', '', string.punctuation)), dataset)

def _parse_I_RO_response(text, dataset):
    if "A:" in text:
        if "The answer is" in text:
            splitted_text = text.strip().split("The answer is")
            rationale = splitted_text[0].strip()
            label_text = _parse_label_text(splitted_text[1].split(".")[0].strip(), dataset)
        elif "So the answer is" in text:
            splitted_text = text.strip().split("So the answer is")
            rationale = splitted_text[0].strip()
            label_text = _parse_label_text(splitted_text[1].split(".")[0].strip(), dataset)
        else:
            return None, None
    else:
        splitted_text = text.strip().split(".")
        print("_+_+_+_+_+_+_+_+_+_+_+_+_+_++")
        print(splitted_text)
        print("_+_+_+_+_+_+_+_+_+_+_+_+_+_++")
        label_text = _parse_label_text(splitted_text[-2], dataset)
        rationale = " ".join(splitted_text[:-2])
    return label_text, rationale

def parse_response(response_text, gen_mode, incontext, prompt_type, dataset):
    if gen_mode == "I-O":
        if incontext == "cot":
            if prompt_type == "cot":
                label = _parse_label_text(response_text)
                rationale = None
        elif prompt_type == 'feb':
                label = _parse_label_text(response_text)
                rationale = None
    elif gen_mode == "I-RO":
        if prompt_type == "cot":
                label, rationale = _parse_I_RO_response(response_text, dataset)    
    elif gen_mode == "I-OR":
        if prompt_type == "feb":
            splitted_response = response_text.split("Reason:")
            label = output_label_mapper(splitted_response[0].strip(), dataset)
            rationale = splitted_response[1].strip()
    else:
        raise NotImplementedError
    return label, rationale

def output_label_mapper(label, dataset):
    if dataset == "strategyqa":
        if label == "yes" or label == "Yes":
            return "true"
        elif label == "no" or label == "No":
            return "false"
        elif label == "True" or label == "False" or label == "true" or label == "false":
            return label.lower()
        else:
            return NotImplementedError
    elif dataset == "openbookqa":
        if label in ['A', 'B', 'C', 'D']:
            return label
        else:
            return NotImplementedError
    elif dataset == "qed":
        return label
    
def input_label_mapper(label, dataset):
    if dataset == "strategyqa":
        if label == True or label == "True":
            return "Yes"
        elif label == False or label == "False":
            return "No"
        else:
            return NotImplementedError
    else:
        return label
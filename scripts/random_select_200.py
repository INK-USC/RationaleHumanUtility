import json
import random
random.seed(0)
data='../data/strategyqa/raw/strategyqa_processed_test.json'
test_data=json.load(open(data,'r'))
false_data=[]
true_data=[]
for data in test_data:
    answer=data['answer']
    if answer==False:
        false_data.append(data)
    else:
        true_data.append(data)
print(len(true_data))
print(len(false_data))
selected_true=random.choices(true_data,k=100)
selected_false=random.choices(false_data,k=100)
# selected_test_data = random.choices(test_data,k=200)

result_file = '../data/strategyqa/raw/strategyqa_processed_test_200.json'
json.dump(selected_true+selected_false,open(result_file,'w'))
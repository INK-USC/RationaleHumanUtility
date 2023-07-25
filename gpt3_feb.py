import openai
import os
import transformers
import json
import random
import argparse
import time
import tqdm
import string

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

def _parse_label_text(text):
    splitted_string = text.strip().split(" ")[-1]
    return output_label_mapper(splitted_string.translate(str.maketrans('', '', string.punctuation)))

def output_label_mapper(label):
    if label == "yes" or label == "Yes":
        return "true"
    elif label == "no" or label == "No":
        return "false"
    elif label == "True" or label == "False" or label == "true" or label == "false":
        return label.lower()
    else:
        return NotImplementedError

def main(args):
    # openai.api_key=os.getenv('OPENAI_API_KEY')
    openai.api_key='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'
    # openai.api_key='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'
    gpt3_version = args.gpt3_version
    # tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
    # task_description='explain qa '
    # max_output_tokens = args.max_output_tokens
    # max_seq_len=args.max_seq_len
    #generate a question given the context and answer the question
    # demonstration="Generate a question given the context and answer the question.\n\n"+\
    # "context:Julius Caesar had three children.Genghis Khan had sixteen children. Modern geneticists have determined that out of every 200 men today has DNA that can be traced to Genghis Khan.\nquestion:Are more people today related to Genghis Khan than Julius Caesar?\nanswer:True\n"+\
    # "context:The members of The Police were musicians, not law enforcement officers. Only law enforcement officers can perform lawful arrests.\nquestion:Could the members of The Police perform lawful arrests?\nanswer:False\n"+\
    # "context:Grey seals have no ear flaps and their ears canals are filled with wax. Grey seals hear better underwater when their ears open like a valve. Dogs have sensitive ears that can hear as far as a quarter of a mile away.\nquestion:Would a dog respond to bell before Grey seal?\nanswer:True\n"+\
    # "context:Rede Globo is a Brazilian television network. The official language of Brazil is Portuguese.\nquestion:Do the anchors on Rede Globo speak Chinese?\nanswer:False\n"+\
    # "context:The average cost of a US Boeing 737 plane is 1.6 million dollars. Wonder Woman (2017 film) grossed over 800 million dollars at the box office.\nquestion:Is a Boeing 737 cost covered by Wonder Woman (2017 film) box office receipts?\nanswer:True\n"+\
    # "context:Casio is a manufacturer of consumer electronics and watches. Petco is a chain store that sells pet supplies like food, bowls, litter, toys, cages and grooming equipment.\nquestion:Can you buy Casio products at Petco?\nanswer:False\n"
    # demonstration="Generate a question given the context and answer the question.\n\n"+\
    # "context:Julius Caesar had three children. Genghis Khan had sixteen children. Modern geneticists have determined that  out of every 200 men today has DNA that can be traced to Genghis Khan.\nquestion:Are more people today related to Genghis Khan than Julius Caesar?\nanswer:True\n"+\
    # "context:Grey seals have no ear flaps and their ears canals are filled with wax. Grey seals hear better underwater when their ears open like a valve. Dogs have sensitive ears that can hear as far as a quarter of a mile away.\nquestion:Would a dog respond to bell before Grey seal?\nanswer:True\n"+\
    # "context:The average cost of a US Boeing 737 plane is 1.6 million dollars. Wonder Woman (2017 film) grossed over 800 million dollars at the box office.\nquestion:Is a Boeing 737 cost covered by Wonder Woman (2017 film) box office receipts?\nanswer:True\n"+\
    # "context:The primary language spoken in Saint Vincent and the Grenadines is Vincentian Creole. Vincentian Creole is English-based, with elements of French, Antillean Creole, and indigenous South American and African languages.\nquestion:Is the language used in Saint Vincent and the Grenadines rooted in English?\nanswer:True\n"+\
    # "context:Christmas trees are usually pine trees. Pine trees keep their needles all year round.\nquestion:Are Christmas trees dissimilar to deciduous trees?\nanswer:True\n"+\
    # "context:Dragon Ball has 6 TV series, 3 TV specials, and 2 direct to video spinoffs as of 2020. Friday the 13th has 12 movies in the franchise and 1 TV series as of 2020.\nquestion:Does Dragon Ball shows and movies fall short of Friday 13th number of projects?\nanswer:True\n"
    # # #2 false
#     demonstration="Paraphrase the question and answer the question given the context and question:\n\n"+\
# "context:Spartina Patens is a type of cordgrass that grows in salt marshes. Spartina Patens requires a marsh-like environment to thrive. The Sahara Desert is known for being dry and very hot.\nquestion:Can Spartina Patens thrive in the Sahara Desert?\nparaphrase:Would Spartina Patens grow in the Sahara Desert?\nanswer:False\n"+\
# "context:Taco Bell has over 7,072 restaurants as of 2018. Roy Rogers had over 600 restaurants at its peak. Roy Rogers has 48 locations as of 2019.\nquestion:Will more people go in and out of Taco Bell than a Roy Rogers each year?\nparapphrase:Is Taco Bell more popular than Roy Rogers?\nanswer:True\n"+\
# "context:A popular desert is carrot cake. Carrot cake is made with carrots. Carrots are a kind of vegetable.\nquestion:Can dessert be made with vegetables?\nparaphrase:Will vegetables be found in dessert?\nanswer:True\n"+\
# "context:A popular desert is carrot cake. Carrot cake is made with carrots. Carrots are a kind of vegetable.\nquestion:Would ramen be bad for someone with heart failure?\nparaphrase:Will ramen do harm to someone with heart failure\nanswer:True\n"+\
# "context:Oysters are an excellent source of zinc. ADHD is a mental disorder of the neurodevelopmental type characterized by difficulty paying attention. Zinc supplementation has been reported to improve symptoms of ADHD and depression.\nquestion:Should oysters be avoided by people with ADHD?\nparaphrase:Is it bad for people with ADHD eating oysters?\nanswer:True\n"+\
# "context:Starbucks is a coffee shop found in numerous countries including USA, China, and the United Arab Emirates. The United Arab Emirates has a Starbucks in Dubai. Islam is the largest and the official state religion of the United Arab Emirates. Pew Research estimates state that over 76% of the citizens of the United Arab Emirates are Islamic.\nquestion:Do any Islamic dominated countries have a Starbucks?\nparaphrase:Is there any Starbucks in Islamic dominated countries?\nanswer:True\n"
#     demonstration="Generate a similar question and answer the question given context and question\n\n"+\
# "context:The Qwerty layout was originally developed for mechanical typewriters in the 1870s. ENIAC was considered to be the first computer, built in the late 1940s.\nquestion:Did the Qwerty keyboard layout predate computers?\ngenerate:Was the Qwerty layout developed for computers?\nanswer:False\n"+\
# "context:Pimp My Ride is a show where people's cars are upgraded and improved The Pope has a personal vehicle called the Popemobile.\nquestion:Could the Pope be on an episode of Pimp My Ride?\ngenerate:Is the Popemobile a car?\nanswer:True\n"+\
# "context:Colitis is a disease in which the colon becomes inflamed. Many things can trigger colitis, including dairy, alcohol, and caffeine. The kola nut is the fruit of the tropical cola tree that contains caffeine inside.\nquestion:Is it best to avoid kola nuts with colitis?\ngenerate:Is colitis triggered by the kola nut?\nanswer:True\n"+\
# "context:When milk becomes acidic, the water and fats separate from each other. When the water and fats separate in milk, it becomes clumpy and has a bad texture. Lemon is highly acidic.\nquestion:Does Lemon enhance the flavor of milk?\ngenerate:Would milk become clumpy and have a bad texture if it was mixed with lemon?\nanswer:True\n"+\
# "context:Rainbows contain the following colors: red, orange, yellow, green, blue, indigo and violet. The flag of Gabon is green, yellow, and blue.\nquestion:Are flag of Gabon colors found in rainbow?\ngenerate:Is the flag of Gabon a rainbow?\nanswer:False\n"+\
# "context:Buffalo wings are fried chicken wings covered in a spicy sauce. Spicy foods are provided their spice from capsaicin from peppers.\nquestion:If someone loves buffalo wings do they enjoy capsaicin?\ngenerate:Do Buffalo wings have capsaicin?\nanswer:True\n"
#     demonstration="Given a context,rephrase the question and answer it:\n\n"+\
# "context:Spartina Patens is a type of cordgrass that grows in salt marshes. Spartina Patens requires a marsh-like environment to thrive. The Sahara Desert is known for being dry and very hot.\nquestion:Can Spartina Patens thrive in the Sahara Desert?\nrephrase:Would Spartina Patens grow in the Sahara Desert?\nanswer:False\n"+\
# "context:Taco Bell has over 7,072 restaurants as of 2018. Roy Rogers had over 600 restaurants at its peak. Roy Rogers has 48 locations as of 2019.\nquestion:Will more people go in and out of Taco Bell than a Roy Rogers each year?\nrephrase:Is Taco Bell more popular than Roy Rogers?\nanswer:True\n"+\
# "context:A popular desert is carrot cake. Carrot cake is made with carrots. Carrots are a kind of vegetable.\nquestion:Can dessert be made with vegetables?\nrephrase:Will vegetables be found in dessert?\nanswer:True\n"+\
# "context:A popular desert is carrot cake. Carrot cake is made with carrots. Carrots are a kind of vegetable.\nquestion:Would ramen be bad for someone with heart failure?\nrephrase:Will ramen do harm to someone with heart failure\nanswer:True\n"+\
# "context:Oysters are an excellent source of zinc. ADHD is a mental disorder of the neurodevelopmental type characterized by difficulty paying attention. Zinc supplementation has been reported to improve symptoms of ADHD and depression.\nquestion:Should oysters be avoided by people with ADHD?\nrephrase:Is it bad for people with ADHD eating oysters?\nanswer:True\n"+\
# "context:Starbucks is a coffee shop found in numerous countries including USA, China, and the United Arab Emirates. The United Arab Emirates has a Starbucks in Dubai. Islam is the largest and the official state religion of the United Arab Emirates. Pew Research estimates state that over 76% of the citizens of the United Arab Emirates are Islamic.\nquestion:Do any Islamic dominated countries have a Starbucks?\nrephrase:Is there any Starbucks in Islamic dominated countries?\nanswer:True\n"
#     demonstration="Generate a similar question to the given question and answer the question.\n\n"+\
# "question:Did the Qwerty keyboard layout predate computers?\nsimilar:Was the Qwerty layout developed for computers?\nanswer:False\n"+\
# "question:Could the Pope be on an episode of Pimp My Ride?\nsimilar:Is the Popemobile a car?\nanswer:True\n"+\
# "question:Is it best to avoid kola nuts with colitis?\nsimilar:Is colitis triggered by the kola nut?\nanswer:True\n"+\
# "question:Does Lemon enhance the flavor of milk?\nsimilar:Would milk become clumpy and have a bad texture if it was mixed with lemon?\nanswer:True\n"+\
# "question:Are flag of Gabon colors found in rainbow?\nsimilar:Is the flag of Gabon a rainbow?\nanswer:False\n"+\
# "question:If someone loves buffalo wings do they enjoy capsaicin?\nsimilar:Do Buffalo wings have capsaicin?\nanswer:True\n"

#     demonstration="Rephrase the question and answer it:\n\n"+\
# "question:Can Spartina Patens thrive in the Sahara Desert?\nrephrase:Would Spartina Patens grow in the Sahara Desert?\nanswer:False\n"+\
# "question:Will more people go in and out of Taco Bell than a Roy Rogers each year?\nrephrase:Is Taco Bell more popular than Roy Rogers?\nanswer:True\n"+\
# "question:Can dessert be made with vegetables?\nrephrase:Will vegetables be found in dessert?\nanswer:True\n"+\
# "question:Would ramen be bad for someone with heart failure?\nrephrase:Will ramen do harm to someone with heart failure\nanswer:True\n"+\
# "question:Should oysters be avoided by people with ADHD?\nrephrase:Is it bad for people with ADHD eating oysters?\nanswer:True\n"+\
# "question:Do any Islamic dominated countries have a Starbucks?\nrephrase:Is there any Starbucks in Islamic dominated countries?\nanswer:True\n"
 
    # demonstration="Paraphrase the question and answer the question given the context and question.\n\n"+\
    # "context:Julius Caesar had three children. Genghis Khan had sixteen children. Modern geneticists have determined that  out of every 200 men today has DNA that can be traced to Genghis Khan.\nquestion:Are more people today related to Genghis Khan than Julius Caesar?\nparaphrase:Do more people today have connection with Genghis Khan than Julius Caesar?\nanswer:True\n"+\
    # "context:Grey seals have no ear flaps and their ears canals are filled with wax. Grey seals hear better underwater when their ears open like a valve. Dogs have sensitive ears that can hear as far as a quarter of a mile away.\nquestion:Would a dog respond to bell before Grey seal?\nparaphrase: Would Grey seal respond to bell later than a dog?\nanswer:True\n"+\
    # "context:The average cost of a US Boeing 737 plane is 1.6 million dollars. Wonder Woman (2017 film) grossed over 800 million dollars at the box office.\nquestion:Is a Boeing 737 cost covered by Wonder Woman (2017 film) box office receipts?\nparaphrase:Does Wonder Woman box office receipts cover a Boeing 737 cost?\nanswer:True\n"+\
    # "context:The primary language spoken in Saint Vincent and the Grenadines is Vincentian Creole. Vincentian Creole is English-based, with elements of French, Antillean Creole, and indigenous South American and African languages.\nquestion:Is the language used in Saint Vincent and the Grenadines rooted in English?\nparaphrase: Does the language used in Saint Vincent and the Grenadines originate from English?\nanswer:True\n"+\
    # "context:Christmas trees are usually pine trees. Pine trees keep their needles all year round.\nquestion:Are Christmas trees dissimilar to deciduous trees?\nparaphrase:Are Christmas trees different from deciduous trees?\nanswer:True\n"+\
    # "context:Dragon Ball has 6 TV series, 3 TV specials, and 2 direct to video spinoffs as of 2020. Friday the 13th has 12 movies in the franchise and 1 TV series as of 2020.\nquestion:Does Dragon Ball shows and movies fall short of Friday 13th number of projects?\nparaphrase:Does Dragon Ball make less shows and movies than Friday 13th?\nanswer:True\n"
    # prompt1="context:When Gauss died in 1855, his brain was preserved for study. Dr.Rudolf Wagner, who studied the brain, found the mass to be slightly above average, and found highly developed convolutions on the brain.\nquestion:"
    # demonstration="Given a context,rephrase the question and answer it.\n\n"+\
    # "context:Julius Caesar had three children. Genghis Khan had sixteen children. Modern geneticists have determined that  out of every 200 men today has DNA that can be traced to Genghis Khan.\nquestion:Are more people today related to Genghis Khan than Julius Caesar?\nrephrase:Do more people today have connection with Genghis Khan than Julius Caesar?\nanswer:True\n"+\
    # "context:Grey seals have no ear flaps and their ears canals are filled with wax. Grey seals hear better underwater when their ears open like a valve. Dogs have sensitive ears that can hear as far as a quarter of a mile away.\nquestion:Would a dog respond to bell before Grey seal?\nrephrase: Would Grey seal respond to bell later than a dog?\nanswer:True\n"+\
    # "context:The average cost of a US Boeing 737 plane is 1.6 million dollars. Wonder Woman (2017 film) grossed over 800 million dollars at the box office.\nquestion:Is a Boeing 737 cost covered by Wonder Woman (2017 film) box office receipts?\nrephrase:Does Wonder Woman box office receipts cover a Boeing 737 cost?\nanswer:True\n"+\
    # "context:The primary language spoken in Saint Vincent and the Grenadines is Vincentian Creole. Vincentian Creole is English-based, with elements of French, Antillean Creole, and indigenous South American and African languages.\nquestion:Is the language used in Saint Vincent and the Grenadines rooted in English?\nrephrase: Does the language used in Saint Vincent and the Grenadines originate from English?\nanswer:True\n"+\
    # "context:Christmas trees are usually pine trees. Pine trees keep their needles all year round.\nquestion:Are Christmas trees dissimilar to deciduous trees?\nrephrase:Are Christmas trees different from deciduous trees?\nanswer:True\n"+\
    # "context:Dragon Ball has 6 TV series, 3 TV specials, and 2 direct to video spinoffs as of 2020. Friday the 13th has 12 movies in the franchise and 1 TV series as of 2020.\nquestion:Does Dragon Ball shows and movies fall short of Friday 13th number of projects?\nrephrase:Does Dragon Ball make less shows and movies than Friday 13th?\nanswer:True\n"
    # demonstration="Rephrase the question and answer it.\n\n"+\
    # "question:Are more people today related to Genghis Khan than Julius Caesar?\nrephrase:Do more people today have connection with Genghis Khan than Julius Caesar?\nanswer:True\n"+\
    # "question:Would a dog respond to bell before Grey seal?\nrephrase: Would Grey seal respond to bell later than a dog?\nanswer:True\n"+\
    # "question:Is a Boeing 737 cost covered by Wonder Woman (2017 film) box office receipts?\nrephrase:Does Wonder Woman box office receipts cover a Boeing 737 cost?\nanswer:True\n"+\
    # "question:Is the language used in Saint Vincent and the Grenadines rooted in English?\nrephrase: Does the language used in Saint Vincent and the Grenadines originate from English?\nanswer:True\n"+\
    # "question:Are Christmas trees dissimilar to deciduous trees?\nrephrase:Are Christmas trees different from deciduous trees?\nanswer:True\n"+\
    # "question:Does Dragon Ball shows and movies fall short of Friday 13th number of projects?\nrephrase:Does Dragon Ball make less shows and movies than Friday 13th?\nanswer:True\n"
    prompt2="context:When milk becomes acidic, the water and fats separate from each other. When the water and fats separate in milk, it becomes clumpy and has a bad texture. Lemon is highly acidic.\nquestion:"
    prompt8="context:Spartina Patens is a type of cordgrass that grows in salt marshes. Spartina Patens requires a marsh-like environment to thrive. The Sahara Desert is known for being dry and very hot.\nquestion:"
    prompt19="context:Beauty and the Beast is a fairy tale adapted into several movie and TV shows.Kurt Sutter created the TV series Sons of Anarchy and The Shield.Charlie Hunnam and Ron Perlman starred in Sons of Anarchy.Ron Perlman starred in the TV series Beauty and the Beast which aired from 1987-1990.\nquestion:"
    prompt20="context:Naruto is a character in a Japanese anime and manga about ninjas.The Temple of Doom is a setting from an Indiana Jones movie\nquestion:"
    prompt21="context:The President of South Korea lives in the Blue House.The Blue House finished construction in early 1991.The World Trade Center complex was completed in 1987\nquestion:"
    prompt22="context:American Universities are known for being liberal in their demographics.Groups like the Ku Klux Klan are condemned by liberal groups, as they advocate for human equality.\nquestion:"
    prompt23="context:The 10th Doctor in David Who is played by David Tennant.In multiple episodes of the series, the 10th doctor mentions that he hates pears.\nquestion:"
    prompt24="context:The Andes includes high, dry zones without precipitation.Dry climates do not impede mummification.Many mummies have been found in the Andes.\nquestion:"
    prompt25="context:Dopamine is a hormone and a neurotransmitter.Neurotransmitters are produced endogenously by the body and are not consumed externally.\nquestion:"
    prompt26="context:Archduke Franz Ferdinand of Austria was assassinated in 1914.The Pacific War took place between 1941 and 1945.\nquestion:"
    prompt27="context:The first Vice President of the United States was John Adams.The Ottomans were a Turkic group that conquered Constantinople in 1453.John Adams was descended from English Puritans.\nquestion:"
    prompt28="context:Steve Carell plays Michael Scott on The Office.Michael Scott is a clueless and naive character that is not meant to be seen as effective in his job as General Manager.\nquestion:"
    prompt29="context:Zorro was known for using his weapon to leave a mark wherever he went.The mark Zorro left was the first initial of his name and nothing more.\nquestion:"
    prompt30="context:All forms of cancer qualify as diagnoses that can result in disability.Disability is not determined by diagnosis, but by degree of impairment.Some cancer patients do not experience major impairment.\nquestion:"
    #2 true
    prompt3="context:People with heart failure have to limit their sodium intake. Ramen is notorious for having incredibly high sodium levels.\nquestion:"
    prompt4="context:Tearjerkers typically refer to a genre of movie. United Airlines flight 93 was involved in a terrorist attack in 2001. Several flights memorialize the passengers of Flight 93,.\nquestion:"
    prompt5="context:Rainbows contain the following colors: red, orange, yellow, green, blue, indigo and violet. The flag of Gabon is green, yellow, and blue.\nquestion:"
    prompt6="context:The Qwerty layout was originally developed for mechanical typewriters in the 1870s. ENIAC was considered to be the first computer, built in the late 1940s.\nquestion:"
    prompt7="context:Taco Bell has over 7,072 restaurants as of 2018. Roy Rogers had over 600 restaurants at its peak. Roy Rogers has 48 locations as of 2019.\nquestion:"
    prompt9="context:Author J.D. Salinger had two children.Actor Anthony Quinn had twelve children.\nquestion:"
    prompt10="context:E.T., the main alien from E.T. the Extra-Terrestrial, loved Reese's Pieces candy.Friendly's is a restaurant that serves dinner entrees and ice cream dishes.Friendly's has several desserts with Reese's candy including the Reese's Peanut Butter Cup Sundae, and Reese's Pieces Sundae.\nquestion:"
    prompt11="context:Based on the American Culinary Federation, the minimum requirements for entering culinary apprenticeships include being 17 years old and having a high school diploma or equivalent.Michael Jordan graduated from Laney High School in 1981.Michael Jordan was born on February 17, 1963, which makes him 57 years old in 2020.\nquestion:"
    prompt12="context:The swastika is a religious symbol that is used in Hinduism, Buddhism, and Jainism.Almost 80% of people in India practice Hinduism.\nquestion:"
    prompt13="context:Christina Maria Aguilera was born on December 18, 1980, in Staten Island, New York.Staten Island has sometimes been called \"the forgotten borough\" by inhabitants who feel neglected by the city government.\nquestion:"
    prompt14="context:Sony is the maker of the Playstation which has sold over 108 million PS4 units by March 2020.Sega's last console, the Sega Dreamcast, was discontinued in 2001.Sony Playstation competed with Sega's Dreamcast and Saturn systems in the 1990s.Sega now makes games for its former competitor, Sony, including Team Sonic Racing in 2019.At the height of the console wars, Sega Saturn sold 9.5 million units while Sony Playstation sold 102 million units.\nquestion:"
    prompt15="context:To be alliterative, words must have the same initial consonant sound.The names of The Powerpuff Girls are Blossom, Buttercup, and Bubbles.\nquestion:"
    prompt16="context:Most international airports and aviators use the foot to measure altitude. China and North Korea require pilots to use meters for altitude. Pilots must communicate their altitude with local air traffic control.\nquestion:"
    prompt17="context:Parts of the Louvre are built of limestone.Nitric acid dissolves limestone.\nquestion:"
    prompt18="context:Mary, Queen of Scots was Queen of Scotland in the 1500s.Mary, Queen of Scots was the daughter of Mary of Guise.Mary of Guise was born to a French nobleman, and her mother was French as well.\nquestion:"
    # No rationales
    # prompt = "Generate a similar question for the given question and rationale. \n\n" + \
    #         "question: Are you likely to find a crucifix in Karachi?\nsimilar question: Are you likely to find a crucifix in Lahore?\n\n" + \
    #         "question: Would a sophist use an épée?\nsimilar question: Would a philantropist use an épée?\n\n" + \
    #         "question: Does penicillin cure a learning disability?\nsimilar question: Does penicillin cure acid attacks?\n\n" + \
    #         "question: If someone is a vegan, would they eat honey?\nsimilar question:  If someone is a vegan, would they eat spinach?\n\n" + \
    #         "question: Is the Matrix a standalone movie?\nsimilar question: Is the Matrix a trilogy?\n\n" + \
    #         "question: If someone loves buffalo wings do they enjoy capsaicin?\nsimilar question: If someone loves desserts do they enjoy capsaicin?\n\n" + \
    #         "question: Does store bought milk have cream at the top?\nsimilar question: "
  

              # "question: If someone is a vegan, would they eat honey?\nrationale: Veganism is a type of diet that excludes all animal products, including those that are byproducts.  Honey is considered an animal byproduct.\nsimilar question:  Can spinach be consumed by vegans?\n\n" 

    # With rationales
    # prompt = "Generate a similar question for the given question and rationale. \n\n" + \
    #         "question: Are you likely to find a crucifix in Karachi?\nrationale: The crucifix is a symbol of Christianity The vast majority of Pakistan's population is Muslim.\nsimilar question: Are you likely to find a crucifix in Lahore?\n\n" + \
    #         "question: Would a sophist use an épée?\nrationale: A sophist is a specific kind of teacher in ancient Greece, in the fifth and fourth centuries BC. Sophists specialized in using the tools of philosophy and rhetoric, though other sophists taught subjects such as music, athletics and mathematics. An épée is a sword used in fencing. The épée was not developed until the 19th century.\nsimilar question: Can an épée be used by a philanthropist?\n\n" + \
    #         "question: Does penicillin cure a learning disability?\nrationale: Learning disabilities are neurological impairments Neurological impairments can result from genetic issues, developmental problems, and accidents like head trauma, malnutrition or exposure to toxins Penicillin is an antibiotic that treats bacterial infection.\nsimilar question: Can penicillin be used to treat acid attack victims?\n\n" + \
    #         "question: Is the Matrix a standalone movie?\nrationale: The Matrix ends in a cliffhanger. The story is then resolved in two sequels, making a trilogy. There are also supplemental works adding to the story, such as a video game and the Animatrix.\nsimilar question: Can standalone movies be parts of trilogies?\n\n" + \
    #         "question: If someone loves buffalo wings do they enjoy capsaicin?\nrationale: Buffalo wings are fried chicken wings covered in a spicy sauce. Spicy foods are provided their spice from capsaicin from peppers.\nsimilar question: Do desserts contain capsaicin?\n\n" 
    # question="question: Did Gauss have a normal brain structure?\nrationale: When Gauss died in 1855, his brain was preserved for study. Dr. Rudolf Wagner, who studied the brain, found the mass to be slightly above average, and found highly developed convolutions on the brain.\nsimilar question:"   
    # question='question: Are LinkedIn and LeafedIn related companies?\nrationale: LinkedIn successfully sued LeafedIn for their choice of name. LeafedIn changed their company name to LeafedOut.\nsimilar question:'
    # question="question: Does Lemon enhance the flavor of milk?\nrationale: When milk becomes acidic, the water and fats separate from each other. When the water and fats separate in milk, it becomes clumpy and has a bad texture.Lemon is highly acidic.\nsimilar question:"
    # question='question: Would ramen be bad for someone with heart failure?\nrationale: People with heart failure have to limit their sodium intake. Ramen is notorious for having incredibly high sodium levels.\nsimilar question:'
    # As question generation with demonstrations
    # prompt = "Generate a question given the context. \n\n" + \
    #         "context: The crucifix is a symbol of Christianity The vast majority of Pakistan's population is Muslim.\nquestion: Are you likely to find a crucifix in Karachi?\n\n" + \
    #         "context: Learning disabilities are neurological impairments Neurological impairments can result from genetic issues, developmental problems, and accidents like head trauma, malnutrition or exposure to toxins Penicillin is an antibiotic that treats bacterial infection.\nquestion: Does penicillin cure a learning disability?\n\n" + \
    #         "context: Veganism is a type of diet that excludes all animal products, including those that are byproducts. Honey is considered an animal byproduct.\nquestion: If someone is a vegan, would they eat honey?\n\n" + \
    #         "context: The Matrix ends in a cliffhanger. The story is then resolved in two sequels, making a trilogy. There are also supplemental works adding to the story, such as a video game and the Animatrix.\nquestion: Is the Matrix a standalone movie?\n\n" + \
    #         "context: Buffalo wings are fried chicken wings covered in a spicy sauce. Spicy foods are provided their spice from capsaicin from peppers.\nquestion: If someone loves buffalo wings do they enjoy capsaicin?\n\n" + \
    #         "context: The Reformation took place in the 16th century. Barack Obama was born in 1961.\nquestion: Did Barack Obama participate in the Reformation?\n\n" 
            # "context: The Qwerty layout was originally developed for mechanical typewriters in the 1870s. ENIAC was considered to be the first computer, built in the late 1940s.\nquestion: "
    # question="context: When milk becomes acidic, the water and fats separate from each other. When the water and fats separate in milk, it becomes clumpy and has a bad texture.Lemon is highly acidic.\nparaphrase:"
    # question= "context: People with heart failure have to limit their sodium intake. Ramen is notorious for having incredibly high sodium levels.\nquestion:"
    # question="context: When milk becomes acidic, the water and fats separate from each other. When the water and fats separate in milk, it becomes clumpy and has a bad texture.Lemon is highly acidic.\nquestion:"
    # question='context: When Gauss died in 1855, his brain was preserved for study. Dr. Rudolf Wagner, who studied the brain, found the mass to be slightly above average, and found highly developed convolutions on the brain.\nquestion:'
    # prompt = "Paraphase the question given the context. \n\n context: When Gauss died in 1855, his brain was preserved for study. Dr. Rudolf Wagner, who studied the brain, found the mass to be slightly above average, and found highly developed convolutions on the brain. question: Did Gauss have a normal brain structure? paraphase: Was brain structure of Gauss normal? \n\n" + \
    #         "context: When milk becomes acidic, the water and fats separate from each other. When the water and fats separate in milk, it becomes clumpy and has a bad texture.Lemon is highly acidic. question: Does Lemon enhance the flavor of milk? paraphase: Will the flavor of milk be enhanced by lemon? \n\n" + \
    #         "context: People with heart failure have to limit their sodium intake. Ramen is notorious for having incredibly high sodium levels. question: Would ramen be bad for someone with heart failure? paraphase: Would ramen harm someone with heart failure? \n\n" + \
    #         "context: LinkedIn successfully sued LeafedIn for their choice of name. LeafedIn changed their company name to LeafedOut. question: Are LinkedIn and LeafedIn related companies? paraphase:"
    # prompt='Paraphase the question given the context. \n\n context: When Gauss died in 1855, his brain was preserved for study. Dr. Rudolf Wagner, who studied the brain, found the mass to be slightly above average, and found highly developed convolutions on the brain. question: Did Gauss have a normal brain structure? paraphase: Was brain structure of Gauss normal? \n\n context: When milk becomes acidic, the water and fats separate from each other. When the water and fats separate in milk, it becomes clumpy and has a bad texture.Lemon is highly acidic. question: Does Lemon enhance the flavor of milk? paraphase: Will the flavor of milk be enhanced by lemon? \n\n context: People with heart failure have to limit their sodium intake. Ramen is notorious for having incredibly high sodium levels. question: Would ramen be bad for someone with heart failure? paraphase: Would ramen harm someone with heart failure? \n\n'
#     demonstration="Given the context, question and answer, generate a similar question that changes the answer, and answer the question. \n\n"+\

# "context: Spartina Patens is a type of cordgrass that grows in salt marshes. Spartina Patens requires a marsh-like environment to thrive. The Sahara Desert is known for being dry and very hot.\nquestion:Can Spartina Patens thrive in the Sahara Desert?\nanswer: False\nchanged question:Would Spartina Patens grow in the Sahara Desert?\nanswer:False\n\n"+\

# "context: Taco Bell has over 7,072 restaurants as of 2018. Roy Rogers had over 600 restaurants at its peak. Roy Rogers has 48 locations as of 2019.\nquestion:Will more people go in and out of Taco Bell than a Roy Rogers each year?\nparapphrase:Is Taco Bell more popular than Roy Rogers?\nanswer:True\n\n"+\

# "context:A popular desert is carrot cake. Carrot cake is made with carrots. Carrots are a kind of vegetable.\nquestion:Can dessert be made with vegetables?\nparaphrase:Will vegetables be found in dessert?\nanswer:True\n\n"+\

# "context:People with heart failure have to limit their sodium intake. Ramen is notorious for having incredibly high sodium levels.\nquestion:Would ramen be bad for someone with heart failure?\nparaphrase:Will ramen do harm to someone with heart failure\nanswer:True\n\n"+\

# "context:Oysters are an excellent source of zinc. ADHD is a mental disorder of the neurodevelopmental type characterized by difficulty paying attention. Zinc supplementation has been reported to improve symptoms of ADHD and depression.\nquestion:Should oysters be avoided by people with ADHD?\nparaphrase:Is it bad for people with ADHD eating oysters?\nanswer:True\n\n"+\

# "context:Starbucks is a coffee shop found in numerous countries including USA, China, and the United Arab Emirates. The United Arab Emirates has a Starbucks in Dubai. Islam is the largest and the official state religion of the United Arab Emirates. Pew Research estimates state that over 76% of the citizens of the United Arab Emirates are Islamic.\nquestion:Do any Islamic dominated countries have a Starbucks?\nparaphrase:Is there any Starbucks in Islamic dominated countries?\nanswer:True\n\n"
#     demonstration="Generate a similar question to the given question and answer the question.\n\n"+\
# "context:A plum tree is a deciduous tree that bears fruit. Deciduous trees shed their leaves in the autumn. Autumn happens from September until the end of Deember.\nquestion:Is November a bad time for a photographer to take pictures of a plum tree in bloom?\ngenerate:Will the leaves a plum tree fall in the autumn?\nanswer:True\n"+\
# "context:The Alamo is located in San Antonio. The Alamo was the site of a major battle during the Texan Revolution against Mexico in 1836.\nquestion:Was San Antonio the site of a major battle in the 19th century?\ngenerate:Was the Alamo the site of a major battle in the 19th century?\nanswer:True\n"+\
# "context:Filicide is the act of killing a son or a daughter. Marvin Gay Sr. committed filicide in 1984 when he shot his son, singer Marvin Gaye. Isaac's father Abraham, was commanded by God to sacrifice his son Isaac, but was spared by an angel.\nquestion:Did Isaac's father almost commit similar crime as Marvin Gay Sr.?\ngenerate:Did Isaac's father almost commit filicide?\nanswer:True\n"+\
# "context:The animals that Yetis are said to look similar to are able to use their hands or toes to grasp items The ability to grasp with hands or other limbs is to be prehensile. \nquestion:Would a Yeti be likely to have prehensile limbs?\ngenerate:Will a Yeti fail to grasp items with its hands or toes?\nanswer:True\n"+\
# "context:Land of Israel was controlled by the Ottoman Empire in 16th century.  The religion of Ottoman Empire was Sunni Islam. \nquestion:Was Land of Israel in possession of an Islamic empire in 16th century?\ngenerate:Was the Ottoman Empire Islamic once?\nanswer:True\n"+\
# "context:Wedding rings are typically made of precious shiny stones such as diamonds. Silicon is a solid rock like element at room temperature that has a natural lustre. Bromine is a liquid at room temperature that is toxic to the touch.\nquestion:Will silicon wedding rings outsell bromine wedding rings?\ngenerate:Are silicon wedding rings shiny?\nanswer:True\n"
#     demonstration="Write a question from the context, that has the same relationship as the previous examples. Examples:\n\n"+\
#     "context:A plum tree is a deciduous tree that bears fruit. Deciduous trees shed their leaves in the autumn. Autumn happens from September until the end of Deember.\nquesion:Is a plum tree in bloom in the autumn?\n"+\
# "context:The animals that Yetis are said to look similar to are able to use their hands or toes to grasp items The ability to grasp with hands or other limbs is to be prehensile. \nquestion:Is a Yeti able to grasp items with its hands or toes?\n"+\
# "context:Keelhauling was a severe punishment whereby the condemned man was dragged beneath the ship\u2019s keel on a rope. Keelhauling is considered a form of torture. Torture is considered cruel. The Eighth Amendment forbids the use of \"cruel and unusual punishment\\nquestion:Would keelhauling be considered cruel?\n"+\
# "context:Khanbaliq was the winter capital of the Mongol Empire.  Khanbaliq was located at the center of what is now modern day Beijing, China. Moon Jae-In was born in Geoje, South Korea.\nquestion:Was Moon Jae-in born in Beijing?\n"+\
# "context:Amazonas is mostly tropical jungle. Tropical jungles contain dangerous creatures. Dangerous creatures put people's lives at risk.\nquestion:Is Amazonas a safe place?\n"+\
# "context:The Los Angeles Memorial Sports Arena had a capacity of 16,740 people. Coachella has had attendance numbers in excess of 99.000 people. Coachella relies on an outdoor set up to accommodate the massive crowds.\nquestion:Would Los Angeles Memorial Sports Arena be too big for Coachella?\n"
#     demonstration="Given the context, question and answer, generate a question that changes the answer.\n\n"+\
#     "context:A plum tree is a deciduous tree that bears fruit. Deciduous trees shed their leaves in the autumn. Autumn happens from September until the end of December.\nquestion:Is November a bad time for a photographer to take pictures of a plum tree in bloom?\nanswer:true\nchanged quesion: Is June a bad time for a photographer to take pictures of a blooming plum tree?\n"+\
# "context:The animals that Yetis are said to look similar to are able to use their hands or toes to grasp items. The ability to grasp with hands or other limbs is to be prehensile. \nquestion:Would a Yeti be likely to have prehensile limbs?\nanswer:true\nchanged question:Would a bird, which does not look similar to Yeti, be prehensile?\n"+\
# "context:Keelhauling was a severe punishment whereby the condemned man was dragged beneath the ship\u2019s keel on a rope. Keelhauling is considered a form of torture. Torture is considered cruel. The Eighth Amendment forbids the use of cruel and unusual punishment. \nquestion:Would keelhauling be a fair punishment under the Eighth Amendment?\nanswer:false\nchanged question:Are cruel punishments forbidden under the Eighth Amendment?\n"+\
# "context:Khanbaliq was the winter capital of the Mongol Empire. Khanbaliq was located at the center of what is now modern day Beijing, China. Moon Jae-In was born in Geoje, South Korea.\nquestion:Was Moon Jae-in born outside of Khanbaliq?\nanswer:true\nchanged question:Was Moon Jae-In born in Beijing?\n"+\
# "context:Amazonas is mostly tropical jungle. Tropical jungles contain dangerous creatures. Dangerous creatures put people's lives at risk.\nquestion:Does walking across Amazonas put a person's life at risk?\nanswer:true\nchanged question:Can a person roam freely in a tropical jungle?\n"+\
# "context:The Los Angeles Memorial Sports Arena had a capacity of 16,740 people. Coachella has had attendance numbers in excess of 99.000 people. Coachella relies on an outdoor set up to accommodate the massive crowds.\nquestion:Was Los Angeles Memorial Sports Arena hypothetically inadequate for hosting Coachella?\nanswer:true\nchanged question:Would everyone who went to Coachells fit into the Los Angeles Memorial Sports Arena?\n" 
#     demonstration="Given the context, question and answer, generate a question that changes the answer. \n\n"+\
# "context:A plum tree is a deciduous tree that bears fruit. Deciduous trees shed their leaves in the autumn. Autumn happens from September until the end of Deember.\nquestion:Is November a bad time for a photographer to take pictures of a plum tree in bloom?\nanswer:true\nchanged quesion:Is a plum tree in bloom in the autumn?\n"+\
# "context:The animals that Yetis are said to look similar to are able to use their hands or toes to grasp items The ability to grasp with hands or other limbs is to be prehensile. \nquestion:Would a Yeti be likely to have prehensile limbs?\nanswer:true\nchanged question:Is a Yeti able to grasp items with its hands or toes?\n"+\
# "context:Keelhauling was a severe punishment whereby the condemned man was dragged beneath the ship\u2019s keel on a rope. Keelhauling is considered a form of torture. Torture is considered cruel. The Eighth Amendment forbids the use of \"cruel and unusual punishment\\nquestion:Would keelhauling be a fair punishment under the Eighth Amendment?\nanswer:false\nchanged question:Would keelhauling be considered cruel?\n"+\
# "context:Khanbaliq was the winter capital of the Mongol Empire.  Khanbaliq was located at the center of what is now modern day Beijing, China. Moon Jae-In was born in Geoje, South Korea.\nquestion:Was Moon Jae-in born outside of Khanbaliq?\nanswer:true\nchanged question:Was Moon Jae-in born in Beijing?\n"+\
# "context:Amazonas is mostly tropical jungle. Tropical jungles contain dangerous creatures. Dangerous creatures put people's lives at risk.\nquestion:Does walking across Amazonas put a person's life at risk?\nanswer:true\nchanged question:Is Amazonas a safe place?\n"+\
# "context:The Los Angeles Memorial Sports Arena had a capacity of 16,740 people. Coachella has had attendance numbers in excess of 99.000 people. Coachella relies on an outdoor set up to accommodate the massive crowds.\nquestion:Was Los Angeles Memorial Sports Arena hypothetically inadequate for hosting Coachella?\nanswer:true\nchanged question:Would Los Angeles Memorial Sports Arena be too big for Coachella?\n"
#     demonstration="Generate a similar question that changes the answer given the question and context. \n\n"+\
# "context:A plum tree is a deciduous tree that bears fruit. Deciduous trees shed their leaves in the autumn. Autumn happens from September until the end of Deember.\nquestion:Is November a bad time for a photographer to take pictures of a plum tree in bloom?\nanswer:true\nchanged quesion:Is a plum tree in bloom in the autumn?\n"+\
# "context:The animals that Yetis are said to look similar to are able to use their hands or toes to grasp items The ability to grasp with hands or other limbs is to be prehensile. \nquestion:Would a Yeti be likely to have prehensile limbs?\nanswer:true\nchanged question:Is a Yeti able to grasp items with its hands or toes?\n"+\
# "context:Keelhauling was a severe punishment whereby the condemned man was dragged beneath the ship\u2019s keel on a rope. Keelhauling is considered a form of torture. Torture is considered cruel. The Eighth Amendment forbids the use of \"cruel and unusual punishment\\nquestion:Would keelhauling be a fair punishment under the Eighth Amendment?\nanswer:false\nchanged question:Would keelhauling be considered cruel?\n"+\
# "context:Khanbaliq was the winter capital of the Mongol Empire.  Khanbaliq was located at the center of what is now modern day Beijing, China. Moon Jae-In was born in Geoje, South Korea.\nquestion:Was Moon Jae-in born outside of Khanbaliq?\nanswer:true\nchanged question:Was Moon Jae-in born in Beijing?\n"+\
# "context:Amazonas is mostly tropical jungle. Tropical jungles contain dangerous creatures. Dangerous creatures put people's lives at risk.\nquestion:Does walking across Amazonas put a person's life at risk?\nanswer:true\nchanged question:Is Amazonas a safe place?\n"+\
# "context:The Los Angeles Memorial Sports Arena had a capacity of 16,740 people. Coachella has had attendance numbers in excess of 99.000 people. Coachella relies on an outdoor set up to accommodate the massive crowds.\nquestion:Was Los Angeles Memorial Sports Arena hypothetically inadequate for hosting Coachella?\nanswer:true\nchanged question:Would Los Angeles Memorial Sports Arena be too big for Coachella?\n"

#     demonstration="Given the context and question, generate a question that negates the question.\n\n"+\
# "context:A plum tree is a deciduous tree that bears fruit. Deciduous trees shed their leaves in the autumn. Autumn happens from September until the end of Deember.\nquestion:Is November a bad time for a photographer to take pictures of a plum tree in bloom?\ngenerate:Is a plum tree in bloom in the autumn?\n"+\
# "context:The animals that Yetis are said to look similar to are able to use their hands or toes to grasp items The ability to grasp with hands or other limbs is to be prehensile. \nquestion:Would a Yeti be likely to have prehensile limbs?\ngenerate:Is a Yeti able to grasp items with its hands or toes?\n"+\
# "context:Keelhauling was a severe punishment whereby the condemned man was dragged beneath the ship\u2019s keel on a rope. Keelhauling is considered a form of torture. Torture is considered cruel. The Eighth Amendment forbids the use of \"cruel and unusual punishment\\nquestion:Would keelhauling be a fair punishment under the Eighth Amendment?\ngenerate:Would keelhauling be considered cruel?\n"+\
# "context:Khanbaliq was the winter capital of the Mongol Empire.  Khanbaliq was located at the center of what is now modern day Beijing, China. Moon Jae-In was born in Geoje, South Korea.\nquestion:Was Moon Jae-in born outside of Khanbaliq?\ngenerate:Was Moon Jae-in born in Beijing?\n"+\
# "context:Amazonas is mostly tropical jungle. Tropical jungles contain dangerous creatures. Dangerous creatures put people's lives at risk.\nquestion:Does walking across Amazonas put a person's life at risk?\ngenerate:Is Amazonas a safe place?\n"+\
# "context:The Los Angeles Memorial Sports Arena had a capacity of 16,740 people. Coachella has had attendance numbers in excess of 99.000 people. Coachella relies on an outdoor set up to accommodate the massive crowds.\nquestion:Was Los Angeles Memorial Sports Arena hypothetically inadequate for hosting Coachella?\ngenerate:Would Los Angeles Memorial Sports Arena be too big for Coachella?\n"
    demonstration="Given a context, generate a similar question to the given question and answer it.\n\n"+\
"context:A plum tree is a deciduous tree that bears fruit. Deciduous trees shed their leaves in the autumn. Autumn happens from September until the end of Deember.\nquestion:Is November a bad time for a photographer to take pictures of a plum tree in bloom?\nsimilar:Will the leaves a plum tree fall in the autumn?\nanswer:True\n"+\
"context:The Alamo is located in San Antonio. The Alamo was the site of a major battle during the Texan Revolution against Mexico in 1836.\nquestion:Was San Antonio the site of a major battle in the 19th century?\nsimilar:Was the Alamo the site of a major battle in the 19th century?\nanswer:True\n"+\
"context:Filicide is the act of killing a son or a daughter. Marvin Gay Sr. committed filicide in 1984 when he shot his son, singer Marvin Gaye. Isaac's father Abraham, was commanded by God to sacrifice his son Isaac, but was spared by an angel.\nquestion:Did Isaac's father almost commit similar crime as Marvin Gay Sr.?\ngsimilar:Did Isaac's father almost commit filicide?\nanswer:True\n"+\
"context:The animals that Yetis are said to look similar to are able to use their hands or toes to grasp items The ability to grasp with hands or other limbs is to be prehensile. \nquestion:Would a Yeti be likely to have prehensile limbs?\nsimilar:Will a Yeti fail to grasp items with its hands or toes?\nanswer:True\n"+\
"context:Land of Israel was controlled by the Ottoman Empire in 16th century.  The religion of Ottoman Empire was Sunni Islam. \nquestion:Was Land of Israel in possession of an Islamic empire in 16th century?\nsimilar:Was the Ottoman Empire Islamic once?\nanswer:True\n"+\
"context:Wedding rings are typically made of precious shiny stones such as diamonds. Silicon is a solid rock like element at room temperature that has a natural lustre. Bromine is a liquid at room temperature that is toxic to the touch.\nquestion:Will silicon wedding rings outsell bromine wedding rings?\nsimilar:Are silicon wedding rings shiny?\nanswer:True\n"
#     demonstration="Generate a similar question to the given question and answer it.\n\n"+\
# "question:Is November a bad time for a photographer to take pictures of a plum tree in bloom?\nsimilar:Will the leaves a plum tree fall in the autumn?\nanswer:True\n"+\
# "question:Was San Antonio the site of a major battle in the 19th century?\nsimilar:Was the Alamo the site of a major battle in the 19th century?\nanswer:True\n"+\
# "question:Did Isaac's father almost commit similar crime as Marvin Gay Sr.?\ngsimilar:Did Isaac's father almost commit filicide?\nanswer:True\n"+\
# "question:Would a Yeti be likely to have prehensile limbs?\nsimilar:Will a Yeti fail to grasp items with its hands or toes?\nanswer:True\n"+\
# "question:Was Land of Israel in possession of an Islamic empire in 16th century?\nsimilar:Was the Ottoman Empire Islamic once?\nanswer:True\n"+\
# "question:Will silicon wedding rings outsell bromine wedding rings?\nsimilar:Are silicon wedding rings shiny?\nanswer:True\n"
    results=[]
    data_dir="../data/strategyqa/raw/strategyqa_processed_test.json"
    test_data=json.load(open(data_dir,'r'))
    for i,inst in enumerate(test_data):
       
        question=inst['question']
      
        context=' '.join(inst['facts'])
        answer="true" if inst["answer"]==True else "false"
        prompt="context:"+context+"\nquestion:"+question+"\nsimilar:"
        # response = openai.Completion.create(engine=gpt3_version, frequency_penalty=0,presence_penalty=0,stop='###',echo=True,logprobs=0,prompt=demonstration+prompt, max_tokens=1024, temperature=0,n=5)
        response = openai.Completion.create(engine=gpt3_version, prompt=demonstration+prompt, max_tokens=1024, temperature=0.7,n=5)
  
        res=response['choices']
        print(question)
        result=''
        for one_res in res:
            result+=one_res['text']+'\n'
        results.append({
            'prompt':prompt,
            'ori_q':question,
            'result':result
        })
        if i>50:
            break
        time.sleep(10)
    json.dump(results,open('similar_question_50_multiple_response_0.7.json','w'))
    

    # res = response['choices'][0]['text']

    # splitted_text = res.strip().split(".")
    # label_text = _parse_label_text(splitted_text[-2])
    # rationale = " ".join(splitted_text[:-2])

    # print(res)
    # print("--------------")
    # print(label_text)
    # print("--------------")
    # print(rationale)

    # total_num_tokens = []
    # output_num_tokens = []
    # count_correct_predictions = 0
    # test_prompts=load_data(args.test_file)
    # train_prompts=load_data(args.train_file)
    # results=[]

    # for index,(test_instance_input,test_instance_output) in enumerate(test_prompts):
        
    #     random.shuffle(train_prompts)
    #     header=task_description
    #     current_seqlen=len(tokenizer.tokenize(f'{header}{test_instance_input}')) + max_output_tokens
    #     for i,(train_instance_input,train_instance_output) in enumerate(train_prompts):
    #         train_instance_seqlen= len(tokenizer.tokenize(f'{train_instance_input}{train_instance_output}'))
    #         if current_seqlen+train_instance_seqlen>max_seq_len:
    #             break
    #         header=header+train_instance_input+train_instance_output
    #         current_seqlen += train_instance_seqlen
    #     prompt = f'{header}{test_instance_input}'
    #     output_num_tokens.append(len(tokenizer.tokenize(test_instance_output)))
    #     total_num_tokens.append(len(tokenizer.tokenize(prompt)) + max_output_tokens)
    #     print(f'{i}-{current_seqlen}-{test_instance_input}{test_instance_output}')
    #     response = openai.Completion.create(engine=gpt3_version, prompt=prompt, max_tokens=max_output_tokens, temperature=0.0)
    #     response_text = response['choices'][0]['text']
    #     predicted_answer=response_text.split()[0][:-1]
    #     ground_truth_answer=test_instance_output.split()[0][:-1]
    #     if predicted_answer==ground_truth_answer:
    #         count_correct_predictions+=1
    #     result={'prompt':prompt,'response_text':response_text}
    #     results.append(result)
        
    #     if index%59==0:
    #         time.sleep(60)

    # print(len(results))
        
    # print('the accuracy is ', count_correct_predictions/len(test_prompts))
    # print(count_correct_predictions)
    # f=open(args.output_file,'w')
    # json.dump(results,f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessing.')
    # parser.add_argument('--data_dir', default='',type=str)
    # parser.add_argument('--test_file', type=str)
    # parser.add_argument('--train_file', type=str)
    # parser.add_argument('--output_file', type=str)
    parser.add_argument('--gpt3_version', type=str, default='davinci-instruct-beta')
    # parser.add_argument('--max_output_tokens',type=int,default=75)
    # parser.add_argument('--max_seq_len', type=int, default=1024)
    
    args = parser.parse_args()
    main(args)
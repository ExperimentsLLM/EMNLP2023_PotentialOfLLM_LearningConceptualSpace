


import logging
from math import exp
from typing import List
from tqdm import tqdm
from statistics import mean
from time import sleep
from scipy.stats import spearmanr
import pandas as pd 
import openai




class OpenAI:
    """ Language Model. """

    def __init__(self, api_key: str, model: str, sleep_time: int = 3):
        """ Language Model.
        @param api_key: OpenAI API key.
        @param model: OpenAI model.
        """
        logging.info(f'Loading Model: `{model}`')
        openai.api_key = api_key
        self.model = model
        self.sleep_time = sleep_time

    def get_perplexity(self, input_texts: str or List, *args, **kwargs):
        """ Compute the perplexity on recurrent LM.
        :param input_texts: A string or list of input texts for the encoder.
        :return: A value or list of perplexity.
        """
        #print(input_texts)
        single_input = type(input_texts) == str
        input_texts = [input_texts] if single_input else input_texts
        nll = []
        for text in tqdm(input_texts):
            # https://platform.openai.com/docs/api-reference/completions/create
            while True:
                try:
                    completion = openai.Completion.create(
                        model=self.model,
                        prompt=text,
                        logprobs=0, # Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens. https://platform.openai.com/docs/api-reference/completions/create#completions/create-logprobs
                        max_tokens=0,
                        temperature=1.0,
                        echo=True  # Echo back the prompt in addition to the completion https://platform.openai.com/docs/api-reference/completions/create#completions/create-echo
                    )
                    break
                except Exception:
                # except openai.error.RateLimitError:
                    if self.sleep_time is None or self.sleep_time == 0:
                        logging.exception('OpenAI internal error')
                        exit()
                    logging.info(f'Rate limit exceeded. Waiting for {self.sleep_time} seconds.')
                    sleep(self.sleep_time)   
            #nll.append(sum([i for i in completion['choices'][0]['logprobs']['token_logprobs'] if i is not None])) 
            #print(completion['choices'][0]['logprobs'])
            nll=(exp(completion['choices'][0]['logprobs']['token_logprobs'][-1]))
            #print(completion['choices'][0]['logprobs']['token_logprobs'])
            #nll=exp(nll1+nll2)
            #ppl = [(i) for i in nll]
            #print(ppl)
        return nll if single_input else ppl
        
        
        
 #Calculate perplexity.
scorer = OpenAI(api_key="OpenAI-API-KEY", model="text-ada-001")
file = open("foodTitle_list.txt","r")

for token in file:
    scores = scorer.get_perplexity(
       
        #input_texts= "{[food item]} tastes [property:sweet/sour/salty/bitter/umami/fatty]".format(token.strip()) --Prompt1
        #input_texts= "It is known that {[food item]} tastes [property:sweet/sour/salty/bitter/umami/fatty]".format(token.strip()) --Prompt2
)
    
    with open('ada_Sweet.txt', 'a') as f: 
        f.write(str(scores)+'\n')

    
   #Spearman's correlation

f1 = pd.read_csv("baseline_sweetness.txt")
f2 = pd.read_csv("ada_Sweet.txt")

# calculate spearman's correlation
coef, p = spearmanr(f1, f2)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
 print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
else:
 print('Samples are correlated (reject H0) p=%.3f' % p)
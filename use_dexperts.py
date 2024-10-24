import torch
from transformers import AutoTokenizer
from transformers import GemmaTokenizer, GemmaTokenizerFast
from dexperts import DExpertsLlama
from datasets import load_dataset
import argparse
import sacrebleu
import json
from tqdm import tqdm
def process_document(file_path):
    # Open the file and read the content
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split the content based on the assumption that there's a blank line between documents
    documents = content.strip().split('\n\n')
    
    # For each document, we'll join the lines (sentences) and return a list of processed documents
    processed_documents = [' '.join(doc.split('\n')) for doc in documents]

def get_trans_prompt(p):
    messages = [{'role': 'system',
                 'content': 'You are a good translator.'
                 }]
    messages.append({'role': 'user',
                 'content': 'You need to translate the input German sentence to English. Input: {} Please directly reply with the translation, start with "Translation:"'.format(p),
                 })
    return messages

def main():
    
    args = argparse.ArgumentParser()
    args.add_argument('--model_path', type=str, default='/nas/shared/NLP_A100/hf_hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819')
    args.add_argument('--input_file', type=str, default='hgissbkh/WMT23-Test')
    args.add_argument('--model_name', type=str, default='gemma-2-9b-it')
    args.add_argument('--longest_n', type=int, default=40)
    args.add_argument('--batch_size', type=int, default=1)
    args.add_argument('--max_new_tokens', type=int, default=1000)
    args.add_argument('--prefix_length', type=int, default=10)
    args.add_argument('--alpha', type=float, default=0.1)
    args.add_argument('--max_length', type=int, default=512)
    args.add_argument('--top_p', type=float, default=0.9)
    args.add_argument('--temperature', type=float, default=1.0)
    args = args.parse_args()

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    # processed_documents = process_document(args.input_file)
    dataset = load_dataset(args.input_file, split='test')
    # Filter the dataset to only include 'de-en' language pairs
    processed_documents = []
    de_en_dataset = dataset.filter(lambda example: example['lp'] == 'de-en')
    processed_documents = [example['src'] for example in de_en_dataset]
    references = [example['ref'] for example in de_en_dataset]
    model_name = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define chat template
    trans_prompts = []
    # select longest 30 sentences in processed_documents and their ids in processed_documents
    longest_documents = sorted(processed_documents, key=len, reverse=True)[:args.longest_n]
    # print(len(processed_documents))
    longest_documents_ids = [processed_documents.index(doc) for doc in longest_documents]
    longest_references = [references[i] for i in longest_documents_ids]
    for text in longest_documents:
        text = get_trans_prompt(text)
        if isinstance(tokenizer, GemmaTokenizer) or isinstance(tokenizer, GemmaTokenizerFast):
                    if text[0]['role'] == 'system':
                        system_prompt = text.pop(0)['content']
                        text[0]['content'] = system_prompt + "\n" + text[0]['content']
        trans_prompts.append(tokenizer.apply_chat_template(text, add_generation_prompt=True, tokenize=False))



    device = torch.device('cuda:0')


    # Initialize DExpertsLlama model
    model = DExpertsLlama(
        expert_model_name_or_path=model_name,
        antiexpert_model_name_or_path=model_name,
        tokenizer=tokenizer,
        alpha=args.alpha,
    )

    # Move model to GPU and use mixed precision
    model.expert.to(device)
    model.antiexpert.to(device)
    batch_size = args.batch_size
    outputs = []
    with torch.cuda.amp.autocast():
        for i in tqdm(range(0, len(trans_prompts), batch_size)):
            batch_trans_prompts = trans_prompts[i:i+batch_size]
            inputs= tokenizer(batch_trans_prompts, return_tensors="pt", max_length=args.max_length, padding=True, truncation=True)
            output = model.generate(
                input_ids=inputs['input_ids'].to(device),
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=args.top_p,
                temperature=args.temperature,
                use_cache=True,
                attention_mask=inputs['attention_mask'].to(device),
                prefix_length=args.prefix_length,
            )
            # delete input ids from output:
            #print(output.size())
            output = output[:, inputs['input_ids'].size(1):]
            for item in output:
                outputs.append(tokenizer.decode(item, skip_special_tokens=True))
                print(outputs[-1])
                #print(outputs[-1])
    
    sentence_bleu_scores = []
    for translation, reference in zip(outputs, longest_references):
        bleu_score = sacrebleu.sentence_bleu(translation, [reference])
        sentence_bleu_scores.append(bleu_score.score)
    translated_token_count = sum([len(tokenizer(translation).input_ids) for translation in outputs])
    ref_token_count = sum([len(tokenizer(reference).input_ids) for reference in longest_references])
    bleu_overall = sum(sentence_bleu_scores) / len(sentence_bleu_scores)
    print('bleu overall', bleu_overall)
    ratio = translated_token_count / ref_token_count
    print('ratio', ratio)
    results = {
        'bleu_overall': bleu_overall,
        'ratio': ratio,
        'model': args.model_name,
        'alpha': args.alpha,
        'top_p': args.top_p,
        'temperature': args.temperature,
        'max_new_tokens': args.max_new_tokens,
        'prefix_length': args.prefix_length,
        'max_length': args.max_length
    }
    with open('results/{}-alpha-{}-temp-{}-prefix-{}-longest-{}.json'.format(args.model_name, args.alpha, args.temperature, args.prefix_length, args.longest_n), 'w') as f:
        json.dump(results, f)
    

if __name__ == "__main__":
    main()




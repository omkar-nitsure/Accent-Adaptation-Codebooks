#!/bin/python
import editdistance as ed
import pandas as pd

filename="result.wrd.txt"
global_lang = []
global_ref=[]
global_hyp=[]

data = pd.read_csv('/raid/speech/omkar/Accented_ASR/finetuning/accent_splits/csvs/test.tsv',sep='\t')
# data = pd.read_csv('/home/darshanp/datasets/commonvoice/with_indian/test_small.tsv',sep='\t')

data = data[['client_id', 'accent']]
data = data[data['accent'].notna()]

cnt = 0
ignore_next = False
for w in open(filename, 'r').readlines():
    if ('REF:' in w):
        if not ignore_next:
            global_ref.append(' '.join(w.replace('*','').split()[1:]).lower())
    elif("HYP:" in w):
        if not ignore_next:
            global_hyp.append(' '.join(w.replace('*','').split()[1:]).lower())
    elif("id" in w):
        _,id = w.split(" ")
        id = id.split("-")
        id = id[0][1:]
        filtered = data[data["client_id"] == id].to_numpy()
        accent = filtered[0][1]
        if accent not in ['other','bermuda','southatlandtic']:
            global_lang.append(accent)
            cnt +=1 
            ignore_next = False            
        else:
            ignore_next = True

assert(len(global_ref) == len(global_hyp) and len(global_hyp) == len(global_lang))

results = {}

for lang in pd.unique(data['accent']):
    
    if lang in ['other','bermuda','southatlandtic']:
        continue

    ref,hyp = [] , []

    for r,h,l in zip(global_ref,global_hyp,global_lang):
        if l == lang:
            ref.append(r)
            hyp.append(h)
    ed_char=[]
    len_char=[]

    ed_wrd=[]
    len_wrd=[]

    for i in range(len(ref)):
        ed_char.append(ed.eval(''.join(ref[i].split()),''.join(hyp[i].split())))
        ed_wrd.append(ed.eval(ref[i].split(),hyp[i].split()))

        len_char.append(len(''.join(ref[i].split())))
        len_wrd.append(len(ref[i].split()))
        # import pudb; pu.db

    try:
        print(f"{lang} CER : {1.0*sum(ed_char)/sum(len_char):.4f} , WER : {1.0*sum(ed_wrd)/sum(len_wrd):.4f}")
        results[lang] = {'cer':1.0*sum(ed_char)/sum(len_char),'wer':1.0*sum(ed_wrd)/sum(len_wrd)}
    except:
        pass

seen_langs = ['us','england','australia', 'canada', 'scotland']
unseen_langs = [lang for lang in results.keys() if lang not in seen_langs]
print(sorted(seen_langs),end=",")
print(sorted(unseen_langs))

for lang in sorted(seen_langs):
    print(f"{100.0*results[lang]['cer']:.1f},{100.0*results[lang]['wer']:.2f}",end=",")
for lang in sorted(unseen_langs):
    print(f"{100.0*results[lang]['cer']:.2f},{100.0*results[lang]['wer']:.2f}",end=",") 
    

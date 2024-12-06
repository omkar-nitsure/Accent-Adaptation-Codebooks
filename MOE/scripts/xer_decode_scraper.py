#!/usr/bin/env python3
'''
This script prints out the overall WER/CER if two files are pointed to namely as ref and hyp.
It assumes that there is line to line correspondence between the two files and does not check for utt id.
@vinit
'''

import editdistance as ed
import argparse, sys
from tqdm import tqdm
from io import StringIO


def calc_errs(ref_file, hyp_file, out_file):
    err_wrds = 0
    num_wrds = 0

    err_chars = 0
    num_chars = 0

    # Read refs and hyps from file
    refs = [x.strip() for x in open(ref_file, 'r').readlines()]
    hyps = [x.strip() for x in open(hyp_file, 'r').readlines()]
    try:
        assert len(refs) == len(hyps)
    except:
        print("Both files contain unequal number of lines")
    if(out_file):
        output = open(out_file,'w')
    else:
        # Dummy output file if no file provided
        output = StringIO()
    for ref, hyp in zip(refs,hyps):
        tmp_err_wrds = ed.eval(ref.strip().split(),hyp.strip().split())
        tmp_err_chrs = ed.eval(ref.replace(' ',''),hyp.strip().replace(' ',''))
        tmp_num_wrds = len(ref.strip().split())
        tmp_num_chrs = len(ref.strip().replace(' ',''))
        try:
            assert (tmp_num_wrds>0) and (tmp_num_chrs>0)
        except:
            print("Skipping utterance as length is 0")
            continue
        output.write("REF: "+ref.strip()+'\n')
        # output.write(f"REF: {ref.strip()} \n")
        output.write("HYP: "+hyp.strip()+'\n')
        output.write("WER: "+str(1.0*tmp_err_wrds/tmp_num_wrds)+'\n')
        output.write("CER: "+str(1.0*tmp_err_chrs/tmp_num_chrs)+'\n\n\n')

        err_wrds += tmp_err_wrds
        err_chars += tmp_err_chrs
        num_wrds += tmp_num_wrds
        num_chars += tmp_num_chrs



    try:
        assert (num_wrds > 0 and num_chars > 0)
    except:
        print("Divide by zero error. Please check.\n")
        print("WER calc-> "+str(err_wrds)+" word errors and "+str(num_wrds)+" words")
        print("WER calc-> "+str(err_chars)+" word errors and "+str(num_chars)+" words")
        return 
    print("CER is -> "+str(1.0*err_chars/num_chars))
    print("WER is -> "+str(1.0*err_wrds/num_wrds)+"\n")
    output.write('____________________\n\n')
    output.write("Total WER is -> "+str(1.0*err_wrds/num_wrds)+"\n")
    output.write("Total CER is -> "+str(1.0*err_chars/num_chars)+"\n")
    output.close()


# calc_errs("ref.wrd.trn","hyp.wrd.trn", None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enter ref and hyp files. Each line must have ref/hyp only and no utt-ids. Order of utt id is assumed to be the same for both files.",add_help=False)
    parser.add_argument("--refs","-r",type=str,required=True,help="Path for refs file.")
    parser.add_argument("--hyps","-h",type=str,required=True,help="Path for hyps file.")
    parser.add_argument("--out-file","-o",type=str,help="Optional path for output file. If blank, only aggregate is printed",default=None)
    if (len(sys.argv))<5:
        parser.print_help()
    args = parser.parse_args()
    calc_errs(args.refs, args.hyps, args.out_file)

# #!/bin/python
# import editdistance as ed

# filename="result.wrd.txt"
# ref=[]
# hyp=[]

# for w in open(filename, 'r').readlines():
#     if (">>" in w):
#         continue
#     if ('REF:' in w):
#         ref.append(' '.join(w.replace('*','').split()[1:]).lower())
#     elif("HYP:" in w):
#         hyp.append(' '.join(w.replace('*','').split()[1:]).lower())

# assert(len(ref) == len(hyp))

# ed_char=[]
# len_char=[]

# ed_wrd=[]
# len_wrd=[]

# for i in range(len(ref)):
#     ed_char.append(ed.eval(''.join(ref[i].split()),''.join(hyp[i].split())))
#     ed_wrd.append(ed.eval(ref[i].split(),hyp[i].split()))

#     len_char.append(len(''.join(ref[i].split())))
#     len_wrd.append(len(ref[i].split()))
#     # import pudb; pu.db


# print("Total examples: ",len(ref))
# print('CER is --->', 1.0*sum(ed_char)/sum(len_char))
# print('WER is --->', 1.0*sum(ed_wrd)/sum(len_wrd))


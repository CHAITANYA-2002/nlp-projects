"""
tagSchemeConverter.py - NER Tag Scheme Converter

Converts between different NER tag representation schemes used in sequence
labeling tasks. Standard CoNLL-format data format is expected (one token
per line, blank lines between sentences).

Supported conversions:
- IOB  -> BIO:   Fixes invalid IOB sequences (I- without preceding B-)
- BIO  -> BIOES: Adds End (E-) and Single (S-) tags for richer boundary info
- BIOES -> BIO:  Simplifies by merging E->I and S->B
- IOB  -> BIOES: Two-step conversion via BIO intermediate

Usage:
    python tagSchemeConverter.py <conversion_type> <input_file> <output_file>
    Example: python tagSchemeConverter.py BIO2BIOES data.bio data.bioes
"""

from __future__ import print_function

import sys


def BIO2BIOES(input_file, output_file):
    """
    Convert BIO tag scheme to BIOES (also called BMES).
    - B- at end of entity or followed by non-I -> S- (Single)
    - I- at end of entity or followed by non-I -> E- (End)
    """
    print("Convert BIO -> BIOES for file:", input_file)
    with open(input_file,'r') as in_file:
        fins = in_file.readlines()
    fout = open(output_file,'w')
    words = []
    labels = []
    for line in fins:
        if len(line) < 3:
            sent_len = len(words)
            for idx in range(sent_len):
                if "-" not in labels[idx]:
                    fout.write(words[idx]+" "+labels[idx]+"\n")
                else:
                    label_type = labels[idx].split('-')[-1]
                    if "B-" in labels[idx]:
                        if (idx == sent_len - 1) or ("I-" not in labels[idx+1]):
                            fout.write(words[idx]+" S-"+label_type+"\n")
                        else:
                            fout.write(words[idx]+" B-"+label_type+"\n")
                    elif "I-" in labels[idx]:
                        if (idx == sent_len - 1) or ("I-" not in labels[idx+1]):
                            fout.write(words[idx]+" E-"+label_type+"\n")
                        else:
                            fout.write(words[idx]+" I-"+label_type+"\n")
            fout.write('\n')
            words = []
            labels = []
        else:
            pair = line.strip('\n').split()
            words.append(pair[0])
            labels.append(pair[-1].upper())
    fout.close()
    print("BIOES file generated:", output_file)



def BIOES2BIO(input_file, output_file):
    """
    Convert BIOES tag scheme back to BIO.
    - E- (End) -> I- (Inside)
    - S- (Single) -> B- (Begin)
    """
    print("Convert BIOES -> BIO for file:", input_file)
    with open(input_file,'r') as in_file:
        fins = in_file.readlines()
    fout = open(output_file,'w')
    words = []
    labels = []
    for line in fins:
        if len(line) < 3:
            sent_len = len(words)
            for idx in range(sent_len):
                if "-" not in labels[idx]:
                    fout.write(words[idx]+" "+labels[idx]+"\n")
                else:
                    label_type = labels[idx].split('-')[-1]
                    if "E-" in labels[idx]:
                        fout.write(words[idx]+" I-"+label_type+"\n")
                    elif "S-" in labels[idx]:
                        fout.write(words[idx]+" B-"+label_type+"\n")
                    else:
                        fout.write(words[idx]+" "+labels[idx]+"\n")     
            fout.write('\n')
            words = []
            labels = []
        else:
            pair = line.strip('\n').split()
            words.append(pair[0])
            labels.append(pair[-1].upper())
    fout.close()
    print("BIO file generated:", output_file)


def IOB2BIO(input_file, output_file):
    """
    Convert IOB tag scheme to BIO.
    In IOB, I- can start an entity. In BIO, entities must start with B-.
    This fixes any I- tag that should be a B- (first token or after O/different type).
    """
    print("Convert IOB -> BIO for file:", input_file)
    with open(input_file,'r') as in_file:
        fins = in_file.readlines()
    fout = open(output_file,'w')
    words = []
    labels = []
    for line in fins:
        if len(line) < 3:
            sent_len = len(words)
            for idx in range(sent_len):
                if "I-" in labels[idx]:
                    label_type = labels[idx].split('-')[-1]
                    if (idx == 0) or (labels[idx-1] == "O") or (label_type != labels[idx-1].split('-')[-1]):
                        fout.write(words[idx]+" B-"+label_type+"\n")
                    else:
                        fout.write(words[idx]+" "+labels[idx]+"\n")
                else:
                    fout.write(words[idx]+" "+labels[idx]+"\n")
            fout.write('\n')
            words = []
            labels = []
        else:
            pair = line.strip('\n').split()
            words.append(pair[0])
            labels.append(pair[-1].upper())
    fout.close()
    print("BIO file generated:", output_file)


def choose_label(input_file, output_file):
    """Extract only the first and last columns from a CoNLL-format file."""
    with open(input_file,'r') as in_file:
        fins = in_file.readlines()
    with open(output_file,'w') as fout:
        for line in fins:
            if len(line) < 3:
                fout.write(line)
            else:
                pairs = line.strip('\n').split(' ')
                fout.write(pairs[0]+" "+ pairs[-1]+"\n")


if __name__ == '__main__':
    '''Convert NER tag schemes among IOB/BIO/BIOES.
        For example: if you want to convert the IOB tag scheme to BIO, then you run as following:
            python tagSchemeConverter.py IOB2BIO input_iob_file output_bio_file
        Input data format is the standard CoNLL 2003 data format.
    '''
    if sys.argv[1].upper() == "IOB2BIO":
        IOB2BIO(sys.argv[2],sys.argv[3])
    elif sys.argv[1].upper() == "BIO2BIOES":
        BIO2BIOES(sys.argv[2],sys.argv[3])
    elif sys.argv[1].upper() == "BIOES2BIO":
        BIOES2BIO(sys.argv[2],sys.argv[3])
    elif sys.argv[1].upper() == "IOB2BIOES":
        IOB2BIO(sys.argv[2],"temp")
        BIO2BIOES("temp",sys.argv[3])
    else:
        print("Argument error: sys.argv[1] should belongs to \"IOB2BIO/BIO2BIOES/BIOES2BIO/IOB2BIOES\"")

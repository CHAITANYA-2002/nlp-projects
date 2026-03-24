"""
get_aligned_bert_emb.py - BERT Subword-to-Token Embedding Alignment

BERT uses WordPiece tokenization which splits words into subword pieces.
For NER (which operates at the token level), we need to map these subword
embeddings back to the original token boundaries.

This script reads BERT's JSON output (from extract_features.py) and produces
token-level embeddings using one of three alignment strategies:

Modes:
    - 'first': Use the embedding of the first subword piece per token
    - 'mean':  Average all subword piece embeddings for each token
    - 'max':   Element-wise maximum across subword piece embeddings

Usage:
    python get_aligned_bert_emb.py -i bert_output.json -m first -r output.emb
"""


import json
import argparse
import os 
import codecs

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="get token level embedding from BERT"
    )
    # input files
    parser.add_argument("--input_file", "-i", type=str, 
                        help="the output json file of BERT, ")
    parser.add_argument("--mode", "-m", type=str, default="first", 
                        help="Using the first mode or arverage all sub_word, [first | mean | max]")
    parser.add_argument("--output_file", "-r", type=str,
                        help="final embedding file , each token embedding is seperated by delimiter")
    parser.add_argument("--delimiter", "-d", type=str, default="|||",
                       help="delimiter of each token embedding in the output file")

    return parser.parse_args(args)


def reduce_mean_list(ls):
    """Compute element-wise average of multiple lists (for 'mean' mode)."""
    if len(ls) == 1:
        return ls[0]
    for item in ls[1:]:
        for index, value in enumerate(item):
            ls[0][index] += value
    return [value / len(ls) for value in ls[0]]

def reduce_max_list(ls):
    """Compute element-wise maximum of multiple lists (for 'max' mode)."""
    if len(ls) == 1:
        return ls[0]
    max_ls = ls[0]
    for item in ls[1:]:
        for index, value in enumerate(item):
            if value > max_ls[index]:
                max_ls[index] = value
    return max_ls


def main(args):
    """
    Read BERT JSON output and produce aligned token-level embeddings.
    Maps subword piece embeddings to original tokens using the specified mode.
    """
    with codecs.open(args.input_file, "r") as input_f, \
         codecs.open(args.output_file, "w", encoding="utf-8") as output_f:
        for line in input_f:
            datas = json.loads(line.strip())
            num_token = len(datas["features"])
            orig_to_tok_map = [id_ for id_ in datas["orig_to_tok_map"] if id_ != 0] + [num_token - 1]
            embeddings = []
            word_pieces_embs = []
            
            for token_id, feature in enumerate(datas["features"]):
                if args.mode == "first" and token_id in orig_to_tok_map[:-1]:
                    embeddings.append(" ".join([str(value) for value in feature["layers"][0]["values"]]))
                    
                if args.mode == "mean" and token_id in orig_to_tok_map[1:]: # merage before word pieces
                    embeddings.append(" ".join([str(value) for value in reduce_mean_list(word_pieces_embs)]))
                    word_pieces_embs = []  # clean word pieces    
                    
                if args.mode == "max" and token_id in orig_to_tok_map[1:]:
                    embeddings.append(" ".join([str(value) for value in reduce_max_list(word_pieces_embs)]))
                    word_pieces_embs = []  

                if token_id > 0 and token_id < num_token - 1: # CLS and SEP are not necessary 
                    word_pieces_embs.append(feature["layers"][0]["values"])   

            output_f.write(args.delimiter.join(embeddings) + "\n")


if __name__ == '__main__':
    main(parse_args())

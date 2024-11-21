#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2020-21: Homework 4
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Vera Lin <veralin@stanford.edu>
"""
import torch
from torch import embedding
import torch.nn as nn
class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layers.

        @param embed_size (int): Embedding size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size
        src_pad_token_idx=vocab.src['<pad>']
        tgt_pad_token_idx=vocab.tgt['<pad>']
        self.source=nn.Embedding(num_embeddings=len(vocab.src),embedding_dim=embed_size,padding_idx=src_pad_token_idx)
        self.target=nn.Embedding(num_embeddings=len(vocab.tgt),embedding_dim=embed_size,padding_idx=tgt_pad_token_idx)
        ### YOUR CODE HERE (~2 Lines)
        ### TODO - Initialize the following variables:
        ###     self.source (Embedding Layer for source language)
        ###     self.target (Embedding Layer for target langauge)
        ###
        ### Note:
        ###     1. `vocab` object contains two vocabularies:
        ###            `vocab.src` for source
        ###            `vocab.tgt` for target
        ###     2. You can get the length of a specific vocabulary by running:
        ###             `len(vocab.<specific_vocabulary>)`
        ###     3. Remember to include the padding token for the specific vocabulary
        ###        when creating your Embedding.
        ###
        ### Use the following docs to properly initialize these variables:
        ###     Embedding Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding
        


        ### END YOUR CODE
        
class MockVocab:
    def __init__(self):
        self.src = {'<pad>': 0, 'hello': 1, 'world': 2}
        self.tgt = {'<pad>': 0, 'hola': 1, 'mundo': 2}

def test_model_embeddings():
    # Use MockVocab instead of a dictionary
    vocab = MockVocab()

    embed_size = 10  # Example embedding size
    embeddings = ModelEmbeddings(embed_size, vocab)

    # Check if embeddings have the correct output shape
    sample_src_input = torch.tensor([1, 2, 0])  # 'hello', 'world', '<pad>'
    sample_tgt_input = torch.tensor([1, 2, 0])  # 'hola', 'mundo', '<pad>'

    src_embedded = embeddings.source(sample_src_input) 
    tgt_embedded = embeddings.target(sample_tgt_input)

    assert src_embedded.shape == (3, embed_size), f"Expected shape (3, {embed_size}), got {src_embedded.shape}"
    assert tgt_embedded.shape == (3, embed_size), f"Expected shape (3, {embed_size}), got {tgt_embedded.shape}"
    
    # Check padding is handled
    assert torch.all(src_embedded[2] == 0), "Padding index for source not set to zero vector"
    assert torch.all(tgt_embedded[2] == 0), "Padding index for target not set to zero vector"

    print("Sanity test passed!")

# Run the sanity test
if __name__ == "__main__":
    test_model_embeddings()        

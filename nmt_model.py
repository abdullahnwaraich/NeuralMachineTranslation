#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2020-21: Homework 4
nmt_model.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
"""
from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from model_embeddings import ModelEmbeddings
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        """ Init NMT Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size, the size of hidden states (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(NMT, self).__init__()
        self.model_embeddings = ModelEmbeddings(embed_size, vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        # In PyTorch, the number of time steps (or sequence length) for the RNN (or LSTM in this case) is determined by the input tensor shape
        # passed to the LSTM during the forward pass, rather than being set in the LSTM definition itself
        self.encoder = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,bidirectional=True,bias=True)
        # the input_size and hidden_size are specified, but the time steps are inferred from the input data shape during the forward pass.
        # When you pass data to self.encoder in the forward function, you typically have an input tensor of shape:
        # input_tensor.shape = (sequence_length, batch_size, input_size) 
        self.decoder = nn.LSTMCell(input_size=embed_size + hidden_size, hidden_size=hidden_size, bias=True)
        # Following layer is a linear transformation applied to the final hidden state of the encoder. It projects the encoder's hidden state 
        # to a vector suitable for initializing the hidden state of the decoder.
        # hidden_size * 2 because our LSTM is bidirectional
        self.h_projection = nn.Linear(2*hidden_size,hidden_size,bias=False) 
        # Following layer is a linear transformation applied to the final cell state of the encoder. It projects the encoder's cell state 
        # to a vector suitable for initializing the cell state of the decoder.
        self.c_projection = nn.Linear(2*hidden_size,hidden_size,bias=False) 
        # Following projection layer is responsible for transforming the encoder’s hidden states (or the attention context vector) 
        # into a compatible size for the attention mechanism. It helps compute the alignment scores by projecting 
        # the encoder’s bidirectional hidden states (dimension hidden_size * 2) to match the hidden_size of the decoder.
        # so we can take the inner products
        self.att_projection = nn.Linear(2*hidden_size,hidden_size,bias=False)
        self.combined_output_projection = nn.Linear(3*hidden_size,hidden_size,bias=False)
        self.target_vocab_projection = nn.Linear(hidden_size,len(vocab.tgt),bias=None)
        self.dropout = nn.Dropout(p=dropout_rate)
        # For sanity check only, not relevant to implementation
        self.gen_sanity_check = True
        self.counter = 0


        ### YOUR CODE HERE (~8 Lines)
        ### TODO - Initialize the following variables:
        ###     self.encoder (Bidirectional LSTM with bias)
        ###     self.decoder (LSTM Cell with bias)
        ###     self.h_projection (Linear Layer with no bias), called W_{h} in the PDF.
        ###     self.c_projection (Linear Layer with no bias), called W_{c} in the PDF.
        ###     self.att_projection (Linear Layer with no bias), called W_{attProj} in the PDF.
        ###     self.combined_output_projection (Linear Layer with no bias), called W_{u} in the PDF.
        ###     self.target_vocab_projection (Linear Layer with no bias), called W_{vocab} in the PDF.
        ###     self.dropout (Dropout Layer)
        ###
        ### Use the following docs to properly initialize these variables:
        ###     LSTM:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        ###     LSTM Cell:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell
        ###     Linear Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
        ###     Dropout Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout




        ### END YOUR CODE


    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)   # Tensor: (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)   # Tensor: (tgt_len, b)

        ###     Run the network forward:
        ###     1. Apply the encoder to `source_padded` by calling `self.encode()`
        ###     2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        ###     3. Apply the decoder to compute combined-output by calling `self.decode()`
        ###     4. Compute log probability distribution over the target vocabulary using the
        ###        combined_outputs returned by the `self.decode()` function.

        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()
        
        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores
    # Recall that, when you pass data to self.encoder in the forward function, you typically have an input tensor of shape:
    # input_tensor.shape = (sequence_length, batch_size, input_size)
    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    
    # Step 1: Embed the source sentences
      X = self.model_embeddings.source(source_padded)  # Shape: (src_len, batch_size, embed_size)

    # Step 2: Pack the embedded sentences for efficient LSTM/GRU processing
      X_packed = torch.nn.utils.rnn.pack_padded_sequence(X, source_lengths, batch_first=False, enforce_sorted=True)

    # Step 3: Pass the packed sequences through the encoder
      enc_hiddens_packed, (last_hidden, last_cell) = self.encoder(X_packed)

    # Step 4: Unpack the hidden states
      enc_hiddens, _ = torch.nn.utils.rnn.pad_packed_sequence(enc_hiddens_packed, batch_first=False)
      enc_hiddens = enc_hiddens.permute(1, 0, 2)  # Convert to shape (batch_size, src_len, hidden_size*2)

    # Step 5: Compute initial decoder hidden and cell states
      last_hidden_concat = torch.cat([last_hidden[0], last_hidden[1]], dim=1)  # Shape: (batch_size, hidden_size*2)
      init_decoder_hidden = self.h_projection(last_hidden_concat)  # Shape: (batch_size, hidden_size)

      last_cell_concat = torch.cat([last_cell[0], last_cell[1]], dim=1)  # Shape: (batch_size, hidden_size*2)
      init_decoder_cell = self.c_projection(last_cell_concat)  # Shape: (batch_size, hidden_size)

    # Step 6: Return encoder hidden states and initial decoder states
      dec_init_state = (init_decoder_hidden, init_decoder_cell)
      return enc_hiddens, dec_init_state

    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
           dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:
    # recall that enc_hiddens.shape=(batch_size,src_len,2*hidden_size)      
    # Chop off the <END> token for max length sentences.
      target_padded = target_padded[:-1]

    # Initialize the decoder state
      dec_state = dec_init_state

    # Initialize the previous combined output vector (o_prev)
      batch_size = enc_hiddens.size(0)
      o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

    # Initialize a list to collect the combined outputs at each step
      combined_outputs = []

    # Precompute the attention projection of encoder hidden states (recall b=batch_size)
      enc_hiddens_proj = self.att_projection(enc_hiddens)  # Shape: (batch_size, src_len, h)

    # Embed the target sentences
      Y = self.model_embeddings.target(target_padded)  # Shape: (tgt_len, b, embed_size)

    # Process each time step
      for Y_t in torch.split(Y, split_size_or_sections=1, dim=0):
        Y_t = Y_t.squeeze(0)  # Shape: (b, embed_size)
        Ybar_t = torch.cat([Y_t, o_prev], dim=1)  # Shape: (b, embed_size + hidden_size)

        # Compute the decoder step
        dec_state, o_t, _ = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)

        # Collect the output
        combined_outputs.append(o_t)
        o_prev = o_t

    # Stack the combined outputs
      combined_outputs = torch.stack(combined_outputs, dim=0)  # Shape: (tgt_len, b, hidden_size)
      return combined_outputs

    def step(self, Ybar_t: torch.Tensor,
         dec_state: Tuple[torch.Tensor, torch.Tensor],
         enc_hiddens: torch.Tensor,
         enc_hiddens_proj: torch.Tensor,
         enc_masks: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
      
      enc_hiddens_proj=self.att_projection(enc_hiddens)
    # Update the decoder state
      dec_state = self.decoder(Ybar_t, dec_state)
      dec_hidden, dec_cell = dec_state  # dec_hidden: (b, h)
    #  print(f"Decoder Hidden State Shape: {dec_hidden.shape}, Cell State Shape: {dec_cell.shape}")  # Debug
    # Compute attention scores
    # self.att_projection(enc_hiddens).shape=(b, src_len, h)
    # dec_hidden.unsqueeze(2).shape=(b,h,1)
      att_scores = torch.bmm(enc_hiddens_proj, dec_hidden.unsqueeze(2)).squeeze(2)
    # att_scores.shape=(batch_size,src_len)  
    #  print(f"Attention Scores Shape (before masking): {att_scores.shape}")  # Debug
    #  if enc_masks is not None:
    #    att_scores.data.masked_fill_(enc_masks.bool(), -float('inf'))
    #    print(f"Attention Scores (after masking): {att_scores}")  # Debug
    # Apply softmax to obtain e_t (attention distribution)
      if enc_masks is not None:
        att_scores.data.masked_fill_(enc_masks.bool(), -float('inf'))

      e_t = F.softmax(att_scores, dim=1)  # Shape: (b, src_len)
    #  print(f"Attention Distribution e_t Shape: {e_t.shape}, Values: {e_t}")  # Debug


    # Compute attention output by applying e_t as weights to enc_hiddens
      att_out = torch.bmm(e_t.unsqueeze(1), enc_hiddens).squeeze(1)  # Shape: (b, 2*h)
    #  print(f"Attention Output att_out Shape: {att_out.shape}, Values: {att_out}")  # Debug

    # Combine dec_hidden and att_out
      U_t = torch.cat((att_out, dec_hidden), dim=1)  # Shape: (b, 3*h)
    #  print(f"Concatenated Tensor U_t Shape: {U_t.shape}")  # Debug
      V_t = self.combined_output_projection(U_t)
      o_t = self.dropout(F.tanh(V_t))  # Shape: (b, h)
      #print(f"Combined Output Shape: {combined_output.shape}, Values: {combined_output}")  # Debug
      return dec_state, o_t, att_scores

    

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size. 
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.
        
        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    def bm_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sent3ence
        """
        src_sents_var=self.vocab.src.to_input_tensor([src_sent],self.device) #src_sents_var.shape=(len[src_sent], batch_size=1) word2indices conversion
        src_encodings, dec_init_vec = self.encode( src_sents_var , [len(src_sent)]) #src_encodings.shape=(1, len[src_sent], h*2)
        src_encodings_att_linear = self.att_projection(src_encodings)
        h_tm1 = dec_init_vec
        att_tm1=torch.zeros(1,self.hidden_size,device=self.device) #(combined-output vector?) Initializes att_tm1, which stores the attention context vector from the previous step.
        eos_id=self.vocab.tgt['</s>']
        hypotheses=[['<s>']] 
        hyp_scores=torch.zeros(len(hypotheses),dtype=torch.float,device=self.device)        
        completed_hypotheses=[]
        t=0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
          t+=1
          hyp_num=len(hypotheses)
          exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))
          exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))
        y_tm1=torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses],dtype=torch.long,device=self.device)
        #Each hypothesis is a list of words (strings) representing the decoded sentence up to the current time step
        y_t_embed=self.model_embeddings.target(y_tm1)
        x=torch.cat([y_t_embed,att_tm1],dim=-1)
        (h_t,cell_t),att_t,_=self.step(x,h_tm1,exp_src_encodings,exp_src_encodings_att_linear,enc_masks=None)
        log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1) #att_t here is combined output vector o_t as P_t=softmax(target_vocab_projection(o_t)) 
        live_hyp_num=beam_size-len(completed_hypotheses)
        contiuating_hyp_scores=(hyp_scores.unsqueeze(1).expand_as(log_p_t)+log_p_t).view(-1)
        
        return src_sent
    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)
        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)
        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)
        eos_id = self.vocab.tgt['</s>']
        hypotheses = [['<s>']] #hypotheses is a List[List[tokens]]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []
        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)
            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))
            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))
            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.model_embeddings.target(y_tm1)
            x = torch.cat([y_t_embed, att_tm1], dim=-1)
            (h_t, cell_t), att_t, _  = self.step(x, h_tm1,
                                                      exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)
            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1) #att_t here is combined output vector o_t as P_t=softmax(target_vocab_projection(o_t)) 
            live_hyp_num = beam_size - len(completed_hypotheses) 
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)
            prev_hyp_ids = top_cand_hyp_pos // len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)
            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []
            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_embeddings.source.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

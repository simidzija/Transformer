"""
Natural language processing tools useful for language model training and 
inference. Includes the following classes:

- Vocab: class for defining a vocabulary given a corpus of text. Provides 
  methods for converting strings to tokens and vice versa.
- TokenizedDataset: class for constructing language model training sets.
- InferenceSampler: class for autoregressive token generation.
"""


import re
import heapq
import torch
from torch.utils.data import Dataset

# custom imports
from transformer import layers
from transformer.layers import Transformer

class Vocab():
    """
    A class for defining a vocabulary, given a corpus. Provides the methods:
        - tokens: converts a string to an int list or tensor
        - words: converts an int list or tensor to a string
        - tokenize_from_file: a wrapper over `tokens` which takes a file path
          as input and produces an int list
    """

    def __init__(self, filepath, pattern):
        self.pattern = pattern
        with open(filepath, mode='r') as file:
            words = re.split(pattern, file.read())
            self.vocab = sorted(set(words))
            self.words_to_tokens = {w: t for t, w in enumerate(self.vocab)}
            self.tokens_to_words = {t: w for t, w in enumerate(self.vocab)}

    def __len__(self):
        return len(self.words_to_tokens)

    def tokens(self, str: str, tensor_device: bool=None) -> list[int] | torch.Tensor:
        """
        Converts a string into an int list or tensor. Returns tensor if 
        tensor_device is given, else returns list.
        """

        words = re.split(self.pattern, str)
        tok_lst = [self.words_to_tokens[word] for word in words]

        if tensor_device is None:
            return tok_lst
        elif isinstance(tensor_device, torch.device):
            return torch.tensor(tok_lst, dtype=torch.long, device=tensor_device)
        else:
            raise TypeError(f'tensor_device must be None or torch.device '
                            f'but got {tensor_device}')

    def words(self, tokens: list[int] | torch.Tensor) -> str:
        """
        Converts an integer list or tensor to a string of words.
        """

        tokens = tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens
        word_lst = [self.tokens_to_words[token] for token in tokens]

        return ''.join(word_lst)

    def tokenize_from_file(self, filepath: str) -> list[int]:
        """
        Wrapper around `tokens` which takes as input a filepath and produces an 
        integer list.
        """
        
        with open(filepath, mode='r') as file:
            toks = self.tokens(file.read())

        return toks
    

class TokenizedDataset(Dataset):
    """
    A class for sectioning off a corpus into fixed size, tokenized training 
    examples. Each training example is a pair (input, target) where input is 
    the same token sequence as target, but right-shifted by one position.
    """

    def __init__(self, corpus: torch.LongTensor, context_window: int, 
                 device: torch.device):
        super().__init__()

        self.corpus = corpus
        self.context_window = context_window
        self.device = device

    def __len__(self):
        return max(1, len(self.corpus) - self.context_window)

    def __getitem__(self, idx):
        assert idx >= 0 and idx < len(self), f'index {idx} out of range'

        input = self.corpus[idx : idx + self.context_window]
        target = self.corpus[idx + 1 : idx + self.context_window + 1]

        return (input, target)
    

class InferenceSampler:
    """
    A class for autoregressive inference sampling according to a specified 
    model and vocabulary. Provides the following methods for generating output 
    text given a prompt string:
        - greedy: outputs most likely next token
        - top_p: for each output postion samples from smallest number of tokens 
          with a cumulative probability greater than p
        - beam_search: searches for highest likelihood output by tracking fixed 
          number of most promising output branches
    """

    def __init__(self, model: Transformer, vocab: Vocab, context_window: int, 
                 device: torch.device):
        self.model = model
        self.vocab = vocab
        self.context_window = context_window
        self.device = device
    
    def greedy(self, prompt: str, num_tokens: int, 
               print_progress: bool=False) -> str:
        """
        Generate num_tokens using greedy sampling.
        """
        self.model.eval()
        
        with torch.no_grad():
            input = self.vocab.tokens(prompt, self.device)
            output = prompt

            # autoregressive token generation loop
            for step in range(num_tokens):

                output_probs = self.model(input, softmax=True) # output probs

                if print_progress:
                    output_toks = torch.argmax(output_probs, dim=-1) 
                    print(f'Step {step:2d} input:  {input.tolist()}')
                    print(f'Step {step:2d} output: {output_toks.tolist()}\n')

                probs = output_probs[-1] # probs of last token
                sample = torch.argmax(probs, dim=-1, keepdim=True) # greedy
                next_word = self.vocab.words(sample) # convert token to word
                output += next_word # append word to output string
                input = torch.cat([input, sample]) # append token to input list
                
            if print_progress:
                output_toks = torch.argmax(output_probs, dim=-1) 
                print(f'Step {step+1:2d} input:  {input.tolist()}')
                print(f'Step {step+1:2d} output: {output_toks.tolist()}')

        self.model.train()

        return output
    
    def top_p(self, prompt: str, num_tokens: int, temp: float=1, 
              p: float=0.9) -> str:
        """
        Generate num_tokens using top_p sampling, with optional temperature. 
        """

        def top_p_dist(probs: torch.Tensor, p: float) -> torch.Tensor:
            """
            Modifies probs dist to contain only top_p samples.
            """
            # sort 
            sorted, indices = probs.sort(descending=True)
            # cumulative sums 
            cumsum = sorted.cumsum(dim=-1)
            # indices of sums greater than p
            idxs = (cumsum >= p).nonzero()
            # index of lowest sum greater than p
            idx = idxs[0,0] if len(idxs) > 0 else None
            # indices of probs tensor which do not contribute to the sum
            discard_idxs = indices[idx + 1:] if idx is not None else []
            # set prob of non contributing indices to zero
            probs[discard_idxs] = 0

        self.model.eval()
        with torch.no_grad():
            input = self.vocab.tokens(prompt, self.device)
            output = prompt

            # autoregressive token generation loop
            for _ in range(num_tokens):
                logits = self.model(input)
                logits /= temp # temperature scaling
                probs = layers.Softmax(-1)(logits)[-1] # full prob dist
                top_p_dist(probs, p) # modifies probs to contain top_p values
                sample = torch.multinomial(probs, 1) # sample from updated probs
                next_word = self.vocab.words(sample) # convert token to word
                output += next_word # append word to output string
                input = torch.cat([input, sample]) # append token to input toks

        self.model.train()
        return output
    
    def beam_search(self, prompt: str, num_tokens: int, 
                    width: float, temp: float=1) -> str:
        """
        Generate num_tokens using beam search, with optional temperature.
        """
        
        self.model.eval()
        input = self.vocab.tokens(prompt, self.device)

        # we will store the beam in a min heap queue, which allows for 
        # efficient retrieval of the beam with the lowest score
        heap = [(0.0, torch.tensor([], dtype=torch.long, device=self.device))]
        
        # autoregressive token generation loop
        for _ in range(num_tokens):

            # create new heap for current inference step
            new_heap = []

            for score, beam in heap:
                
                # concatenate input prompt with current beam output
                beam_input = torch.cat([input, beam])

                # compute model output
                with torch.no_grad():
                    logits = self.model(beam_input)

                # temperature scaling
                logits /= temp 

                # token probability distribution of last output position
                probs = layers.Softmax(-1)(logits)[-1] 

                # width highest probability indices (new branches to explore)
                indices = torch.argsort(probs, descending=True)[:width] 

                # loop over new branches
                for i in indices:

                    # probability of new leaf
                    p = probs[i]

                    # probability of entire beam, including new leaf
                    new_score = score + torch.log(p).item()

                    # push beam to new_heap if new_heap too small
                    if len(new_heap) < width:
                        new_beam = torch.cat([beam, i.unsqueeze(0)])
                        heapq.heappush(new_heap, (new_score, new_beam))

                    # pushpop to new_heap if new_score bigger than lowest score
                    elif new_score > new_heap[0][0]:
                        new_beam = torch.cat([beam, i.unsqueeze(0)])
                        heapq.heappushpop(new_heap, (new_score, new_beam))

            # update heap to new_heap
            heap = new_heap

        # find beam with highest score
        _, max_beam = max(heap, key=lambda x: x[0])

        # convert to string
        output = prompt + self.vocab.words(max_beam)
        
        self.model.train()

        return output
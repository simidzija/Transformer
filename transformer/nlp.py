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
    def __init__(self, filepath, sos, unk, pad, pattern):
        self.pattern = pattern
        self.sos = sos
        self.unk = unk
        self.pad = pad
        self.words_to_tokens = {}
        self.tokens_to_words = {}
        with open(filepath, mode='r') as file:
            token = 0 # next token to be assigned
            for line in file:
                words = re.split(pattern, line, flags=re.IGNORECASE)
                for word in words:
                    if word not in self.words_to_tokens:
                        self.words_to_tokens[word] = token
                        self.tokens_to_words[token] = word
                        token += 1

    def __len__(self):
        return len(self.words_to_tokens)

    def tokens(self, str: str, tensor_device: bool=None, 
               sos: bool=False) -> list[int] | torch.Tensor:
        """
        Converts a string into an int list or tensor. Returns tensor if 
        tensor_device is given, else returns list. Appends start-of-sequence 
        token if sos=True.
        """
        word_lst = re.split(self.pattern, str, flags=re.IGNORECASE)
        tok_lst = [self.sos] if sos else []
        for word in word_lst:
            if word in self.words_to_tokens:
                tok_lst.append(self.words_to_tokens[word])
            else:
                tok_lst.append(self.unk)

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
        word_lst = []
        for token in tokens:
            if token in self.tokens_to_words:
                word_lst.append(self.tokens_to_words[token])
            else:
                raise ValueError(f'{token} not a valid token')

        return ''.join(word_lst)

    def tokenize_from_file(self, filepath: str) -> list[int]:
        """
        Wrapper around `tokens` which takes as input a filepath and produces an 
        integer list.
        """
        
        tokens = []
        with open(filepath, mode='r') as file:
            for line in file:
                tokens.extend(self.tokens(line))

        return tokens
    

class TokenizedDataset(Dataset):
    """
    A class for sectioning off a corpus into fixed size, tokenized, training 
    examples. Each training example is a pair (input, target) where input is 
    the same token sequence as target, but right-shifted by one position to 
    accomodate the start-of-sequence token. 
    """
    def __init__(self, corpus: torch.LongTensor, context_window: int, 
                 sos: int, pad: int, device: torch.device):
        super().__init__()

        self.corpus = corpus
        self.context_window = context_window
        self.sos = sos
        self.pad = pad
        self.device = device

    def __len__(self):
        return max(1, len(self.corpus) - self.context_window + 2)

    def __getitem__(self, idx):
        assert idx >= 0 and idx < len(self), f'index {idx} out of range'

        context = self.corpus[idx:idx + self.context_window - 1]
        input = torch.cat([torch.tensor([self.sos], device=self.device), 
                           context])
        target = torch.cat([context, 
                            torch.tensor([self.pad], device=self.device)])

        return (input, target)
    

class InferenceSampler:
    """
    A class for autoregressive inference sampling according to a specified 
    model and vocabulary. Provides the following methods for generating output 
    text given a prompt string:
        - greedy: outputs most likely next token given previous tokens
        - top_p: for each output postion samples from smallest number of tokens 
          with a cumulative probability greater than p.
    """
    def __init__(self, model: Transformer, vocab: Vocab, context_window: int, 
                 device: torch.device):
        self.model = model
        self.vocab = vocab
        self.context_window = context_window
        self.device = device
    
    def greedy(self, prompt: str, print_progress: bool=False) -> str:
        """
        Outputs most likely next token given previous tokens.
        """
        self.model.eval()
        
        with torch.no_grad():
            input = self.vocab.tokens(prompt, self.device, sos=self.vocab.sos)
            toks_to_generate = self._toks_to_generate(input)
            output = ''
            # autoregressive token generation loop
            for step in range(toks_to_generate):
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
    
    def top_p(self, prompt: str, temp: float=1, p: float=0.9) -> str:
        """
        Perform top_p sampling, with optional temperature. 
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
            idxs = (cumsum >= 0).nonzero()
            # index of lowest sum greater than p
            idx = idxs[0,0] if len(idxs) > 0 else None
            # indices of probs tensor which do not contribute to the sum
            discard_idxs = indices[idx + 1:] if idx is not None else []
            # set prob of non contributing indices to zero
            probs[discard_idxs] = 0

        self.model.eval()
        with torch.no_grad():
            input = self.vocab.tokens(prompt, self.device, sos=self.vocab.sos)
            toks_to_generate = self._toks_to_generate(input)
            output = ''
            # autoregressive token generation loop
            for _ in range(toks_to_generate):
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
    
    def beam_search(self, prompt: str, width: float, temp: float=1, 
                    print_beams: bool=False) -> str:
        """
        Perform beam search output generation, with optional temperature.
        """
        self.model.eval()
        input = self.vocab.tokens(prompt, self.device, sos=self.vocab.sos)

        # we will store the beam in a min heap queue, which allows for 
        # efficient retrieval of the beam with the lowest score
        heap = [(0.0, torch.tensor([], dtype=torch.long, device=self.device))]
        
        # autoregressive token generation loop
        toks_to_generate = self._toks_to_generate(input)
        for _ in range(toks_to_generate):
            # loop over beam heap (use copy so that we can modify original heap)
            heap_copy = heap[:]
            for score, beam in heap_copy:
                beam_input = torch.cat([input, beam])
                with torch.no_grad():
                    logits = self.model(beam_input)
                logits /= temp # temperature scaling
                probs = layers.Softmax(-1)(logits)[-1] # full prob dist
                # highest probability indices - give new branches to explore
                indices = torch.argsort(probs, descending=True)[:width] 
                # loop over new branches
                for i in indices:
                    # compute score
                    p = probs[i]
                    new_score = score - torch.log(p).item()
                    # push to heap if heap is too short
                    if len(heap) < width:
                        new_beam = torch.cat([beam, i.unsqueeze(0)])
                        heapq.heappush(heap, (new_score, new_beam))
                    # pushpop to heap if new_score is bigger than lowest score
                    elif new_score > heap[0][0]:
                        new_beam = torch.cat([beam, i.unsqueeze(0)])
                        heapq.heappushpop(heap, (new_score, new_beam))

        if print_beams:
            print(heap)

        # find beam with highest score
        _, max_beam = max(heap, key=lambda x: x[0])

        # convert to string
        output = self.vocab.words(max_beam)
        
        self.model.train()

        return output


    # ------------------------- Utility functions --------------------------
    
    def _toks_to_generate(self, input):
        """
        Utility function. Given input prompt returns number of tokens to 
        generate before resulting input is too large for context window.
        """
        toks = self.context_window - len(input) + 1
        if toks <= 0:
            raise ValueError(f'prompt (with sos) of length ({len(input)}) ' 
                             f'too large for context window '
                             f'({self.context_window})')
        return toks
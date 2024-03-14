import math, random
import os

################################################################################
# Part 0: Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * n

def ngrams(n, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    ngramList = []
    text = start_pad(n) + text
    for i in range(n, len(text)):
        context = text[i-n:i]
        character = text[i]
        ngramList.append((context, character))
    return ngramList


def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def create_ngram_model_lines(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k):
        self.n = n 
        self.k = k
        self.vocab = set()
        self.ngrams = {}
        self.contexts = {}


    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        padded_text = start_pad(self.n) + text
        for i in range(len(padded_text) - self.n):
            context = padded_text[i:i+self.n]
            char = padded_text[i+self.n]
            self.contexts[context] = self.contexts.get(context, 0) + 1
            ngram = context + char
            self.ngrams[ngram] = self.ngrams.get(ngram, 0) + 1
            self.vocab.add(char)


    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''

        ngram = context + char
        ngram_count = self.ngrams.get(ngram, 0)
        context_count = self.contexts.get(context, 0)
        vocab_size = len(self.vocab)
        
        # Ensure k-smoothing is correctly handled to prevent division by zero
        if self.k > 0:
            prob = (ngram_count + self.k) / (context_count + self.k * vocab_size)
        else:
            # If no smoothing is applied, handle unseen contexts
            if context_count == 0:
                # Return a small non-zero probability for unseen contexts
                return 1 / vocab_size
            else:
                prob = ngram_count / context_count
        
        return prob

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        vocab_list = sorted(list(self.vocab))
        r = random.random()
        cumulative = 0
        for char in vocab_list:
            cumulative += self.prob(context, char)
            if r < cumulative:
                return char
        return vocab_list[-1]


    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        
        result = ''
        context = start_pad(self.n)
        for _ in range(length):
            char = self.random_char(context[-self.n:])
            result += char
            context += char
        return result



    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        log_prob = 0
        for context, char in ngrams(self.n, text):
            prob = self.prob(context, char)
            if prob == 0:
                return float('inf')
            log_prob += math.log(prob, 2)
        return math.pow(2, -log_prob / len(text))

   

################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    class NgramModelWithInterpolation(NgramModel):
        
        def __init__(self, n, k, lambdas=None):
            super().__init__(n, k)
            if lambdas is None:
                self.lambdas = [1/(n+1)] * (n+1)  # Equal weights for simplicity
            else:
                self.lambdas = lambdas

        def prob(self, context, char):
        # Initialize the probability as zero
            probability = 0.0
            
            # Calculate the weighted probability for n-gram order starting from n down to 1 (unigram)
            for i in range(self.n, -1, -1):
                # Extract the n-gram context for the current order
                current_context = context[-i:] if i > 0 else ''
                # Apply add-k smoothing to calculate the probability
                ngram_count = self.ngrams.get(current_context + char, 0) + self.k
                context_count = self.contexts.get(current_context, 0) + self.k * len(self.vocab)
                # Calculate the weighted probability
                weighted_prob = (ngram_count / context_count) if context_count > 0 else 0
                # Update the total probability using the corresponding lambda weight
                probability += self.lambdas[i] * weighted_prob
            
            return probability
    
    def update_lambdas(self, new_lambdas):
        if len(new_lambdas) == 3 and sum(new_lambdas) == 1:
            self.lambdas = new_lambdas
        else:
            raise ValueError("Lambdas must sum to 1 and be a list of three elements.")

        

    def update(self, text):
        # Extend or adapt the existing update method to handle various n-gram lengths
        super().update(text)  # Assuming the superclass method handles basic updating
        
        # You might need to adjust the superclass method or write additional logic here
        # to update counts for n-grams of different lengths if your superclass method doesn't already do this.

    def get_vocab(self):
        # Simply return the vocabulary from the superclass
        return super().get_vocab()

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':

    print("====== Part 0 ======")
    
    print(ngrams(1, 'abc'))
    print(ngrams(2, 'abc'))

    print("====== Part 1 ======")

    m = NgramModel(1,0)
    m.update('abab')
    print("updated vocab with abad")
    print("VOCAB: ", m.get_vocab())
    m.update('abcd')
    print("VOCAB after updating with abcd: ", m.get_vocab())
    print("Prob a, b", m.prob('a', 'b'))
    print("Prob ~, c", m.prob('~', 'c'))
    print("Prob b, c", m.prob('b', 'c'))

    print("Character and Text generation.")
    m = NgramModel(0, 0)
    m.update('abab')
    m.update('abcd')
    # random.seed(1)
    print("Random Characters: ", [m.random_char('') for i in range(25)])
    m = NgramModel(1, 0)
    m.update('abab')
    m.update('abcd')

    print("Random Text: ", m.random_text(25))

    print("====== WRITING SHAKESPEAR ======")
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 2)
    print("1st Try: \n")  
    print(m.random_text(250))
    print("==================================== End ======================================")
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 3)
    print("2nd Try: \n")
    print(m.random_text(250))  
    print("==================================== End ======================================")
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 4)
    print("3rd Try: \n")  
    print(m.random_text(250))  
    print("==================================== End ======================================")
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7)
    print("4th Try: \n") 
    print(m.random_text(250))
    print("==================================== End ======================================")



    print("====== Perplexity ======")


    m = NgramModel(1, 0)  
    m.update('abab') 
    m.update('abcd')  
    print(m.perplexity('abcd'))   
    print(m.perplexity('abca'))  
    print(m.perplexity('abcda'))


    print("====== Smoothing ======")
    m = NgramModel(1, 1)
    print("Updated vocab with abab: ", m.update('abab'))
    print("Updated vocab with abcd: ", m.update('abcd'))
    print("Prob: a,a ", m.prob('a', 'a')) 
    print("Prob:a,b ", m.prob('a', 'b'))
    print("Prob: c, d ", m.prob('c', 'd')) 
    print("Prob: d,a ", m.prob('d', 'a'))



    print("====== Ngram model with interpolation ======")
    m = NgramModelWithInterpolation(1, 0)
    m.update('abab')
    print("a, a", m.prob('a', 'a'))
    print("a, b", m.prob('a', 'b'))

    m = NgramModelWithInterpolation(2, 1)
    m.update('abab')
    m.update('abcd') 
    print("~a, b",m.prob('~a', 'b')) 
    print("ba, b",m.prob('ba', 'b'))
    print("~c, d",m.prob('~c', 'd')) 
    print("~bc, d",m.prob('bc', 'd'))




    folder_path = "./train/"
    af_model = create_ngram_model_lines(NgramModel, "./cities_train/train/af.txt")
    cn_model = create_ngram_model_lines(NgramModel, "./cities_train/train/cn.txt")
    de_model = create_ngram_model_lines(NgramModel, "./cities_train/train/de.txt")
    fi_model = create_ngram_model_lines(NgramModel, "./cities_train/train/fi.txt")
    fr_model = create_ngram_model_lines(NgramModel, "./cities_train/train/fr.txt")
    in_model = create_ngram_model_lines(NgramModel, "./cities_train/train/in.txt")
    pk_model = create_ngram_model_lines(NgramModel, "./cities_train/train/pk.txt")
    za_model = create_ngram_model_lines(NgramModel, "./cities_train/train/za.txt")


    test_data = "./cities_test.txt"
    output_file = "./output_file.txt"
    country_output = ""


    # Open the file
    with open(output_file, 'w', encoding='utf-8') as outputfile:
        with open(test_data, 'r', encoding='utf-8') as file:
            # Iterate over each line in the file
            for line in file:
                # Process the line (strip() removes leading and trailing whitespaces)
                processed_line = line.strip()
                
                af = (af_model.perplexity(processed_line))
                cn = (cn_model.perplexity(processed_line))
                de = (de_model.perplexity(processed_line))
                fi = (fi_model.perplexity(processed_line))
                fr = (fr_model.perplexity(processed_line))
                ind = (in_model.perplexity(processed_line))
                pk = (pk_model.perplexity(processed_line))
                za = (za_model.perplexity(processed_line))


                max_val = min(af, cn, de, fi, fr, ind, pk, za)

                if max_val == af:
                    country_output = "af"
                elif max_val == cn:
                    country_output = "cn"
                elif max_val == de:
                    country_output = "de"
                elif max_val == fi:
                    country_output = "fi"
                elif max_val == fr:
                    country_output = "fr"
                elif max_val == ind:
                    country_output = "in"
                elif max_val == pk:
                    country_output = "pk"
                elif max_val == za:
                    country_output = "za"

                # Write data to the file
                outputfile.write(f"{processed_line}: {country_output}\n")

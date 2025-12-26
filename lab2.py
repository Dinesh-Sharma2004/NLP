#                  POS TAGGING                 #

# 1. Using NLTK
import nltk
from nltk import word_tokenize, pos_tag

sentence = "The apple is red, the sky is blue. Don't try to fit in my shoe!"
tokens = word_tokenize(sentence)

print("NLTK\n")
print(pos_tag(tokens))


# 2. Using SpaCy
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(sentence)

print("\n\nSPACY\n")
for token in doc:
    print(f"{token}:{token.pos_}")


# 3. Using TextBlob
from textblob import TextBlob

text_blob = TextBlob(sentence)
pos_tags = text_blob.tags

print("\n\nTextBlob\n")
print(pos_tags)


# 4. HMM POS Tagging (FROM SCRATCH)
from collections import defaultdict
import math


class HMMPOSTagger:
    def __init__(self):
        self.transition = defaultdict(lambda: defaultdict(int))
        self.emission = defaultdict(lambda: defaultdict(int))
        self.tag_count = defaultdict(int)
        self.tags = set()

    def train(self, tagged_sentences):
        for sentence in tagged_sentences:
            prev_tag = "<START>"
            self.tag_count[prev_tag] += 1

            for word, tag in sentence:
                self.tags.add(tag)
                self.transition[prev_tag][tag] += 1
                self.emission[tag][word] += 1
                self.tag_count[tag] += 1
                prev_tag = tag

            self.transition[prev_tag]["<END>"] += 1

    def transition_prob(self, prev_tag, tag):
        return (self.transition[prev_tag][tag] + 1) / \
               (self.tag_count[prev_tag] + len(self.tags))

    def emission_prob(self, tag, word):
        return (self.emission[tag][word] + 1) / \
               (self.tag_count[tag] + len(self.emission[tag]))

    def viterbi(self, words):
        V = [{}]
        backpointer = [{}]

        # Initialization
        for tag in self.tags:
            V[0][tag] = math.log(self.transition_prob("<START>", tag)) + \
                        math.log(self.emission_prob(tag, words[0]))
            backpointer[0][tag] = "<START>"
            # print(f"\n\n{V}")
            # print(backpointer)

        # Recursion
        for t in range(1, len(words)):
            V.append({})
            backpointer.append({})

            for tag in self.tags:
                # print(max(
                #     (V[t-1][pt] +
                #      math.log(self.transition_prob(pt, tag)) +
                #      math.log(self.emission_prob(tag, words[t])),
                #      pt)
                #     for pt in self.tags
                # ))
                max_prob, best_prev = max(
                    (V[t-1][pt] +
                     math.log(self.transition_prob(pt, tag)) +
                     math.log(self.emission_prob(tag, words[t])),
                     pt)
                    for pt in self.tags
                )

                V[t][tag] = max_prob
                backpointer[t][tag] = best_prev

        # Termination
        best_last_tag = max(V[-1], key=V[-1].get)


        # Backtracking
        tags = [best_last_tag]
        for t in range(len(words)-1, 0, -1):
            tags.insert(0, backpointer[t][tags[0]])
            # print(f"\n\n{tags}")

        return tags



training_data = [
    [("I", "PRON"), ("love", "VERB"), ("NLP", "NOUN")],
    [("You", "PRON"), ("love", "VERB"), ("AI", "NOUN")],
    [("They", "PRON"), ("study", "VERB"), ("NLP", "NOUN")],
    [("Students", "NOUN"), ("study", "VERB"), ("hard", "ADV")],
    [("AI", "NOUN"), ("is", "VERB"), ("powerful", "ADJ")],
    [("NLP", "NOUN"), ("is", "VERB"), ("interesting", "ADJ")]
]

test_sentence = "NLP is powerful"
words = test_sentence.split()

tagger = HMMPOSTagger()
tagger.train(training_data)

predicted_tags = tagger.viterbi(words)

print("\n\nHMM POS TAGGING\n")
for w, t in zip(words, predicted_tags):
    print(f"{w}/{t}")



# 5. Rule Based POS tagging

def rule_based_pos(tokens):
    tags = []
    for i, word in enumerate(tokens):
        if word.lower() in ["the", "a", "an"]:
            tags.append((word, "DT"))
        elif word.lower() in ["new", "good", "high", "special", "big", "local"]:
            tags.append((word, "ADJ"))
        elif word.lower() in ["on", "of", "at", "with", "by", "into", "under"]:
            tags.append((word,"ADP"))
        elif word.lower() in ["really", "already", "still", "early", "now"]:
            tags.append((word,"ADV"))
        elif word.lower()=="is":
            tags.append((word,"VRB"))
        elif word.endswith("ing"):
            tags.append((word, "VBG"))
        elif word.endswith("ed"):
            tags.append((word, "VBD"))
        elif word[0].isupper():
            tags.append((word, "NNP"))
        elif i > 0 and tags[i-1][1] == "DT":
            tags.append((word, "NN"))
        else:
            tags.append((word, "NN"))  
    return tags

sentence = "The apple is running"
tokens = sentence.split()
print("\n\nRule Based Tagging\n")
print(rule_based_pos(tokens))

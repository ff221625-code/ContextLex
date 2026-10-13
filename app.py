from flask import Flask, render_template, request, jsonify
import csv
import os
import re
import socket
from collections import Counter

import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk import word_tokenize, pos_tag

app = Flask(__name__)

# ============================================================
# DATASET
# ============================================================

DATASET = os.path.join(
    os.path.dirname(__file__),
    "dataset",
    "wsd_dataset.csv"
)

wsd_data = []

if os.path.exists(DATASET):
    with open(DATASET, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            wsd_data.append(row)

print("WSD dataset loaded:", len(wsd_data), "sentences")


# ============================================================
# BASIC FUNCTIONS
# ============================================================

STOP = {
    "a", "an", "the", "is", "am", "are", "was", "were",
    "in", "on", "at", "to", "of", "for", "with", "from",
    "under", "over", "near", "out", "up", "and", "or",
    "but", "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "this", "that"
}


def words(text):
    return re.findall(r"[a-zA-Z]+", text.lower())


def clean_context(text, target=""):
    return {
        w for w in words(text)
        if w not in STOP and w != target.lower()
    }


def similarity(sentence, example, target):
    a = clean_context(sentence, target)
    b = clean_context(example, target)

    if not a or not b:
        return 0

    score = len(a & b)

    # Small WordNet similarity bonus
    try:
        for x in a:
            for y in b:
                if x == y:
                    score += 2
                else:
                    sx = wn.synsets(x)
                    sy = wn.synsets(y)

                    for s1 in sx[:2]:
                        for s2 in sy[:2]:
                            sim = s1.wup_similarity(s2)
                            if sim and sim > 0.8:
                                score += 0.3
    except LookupError:
        pass

    return score


# ============================================================
# TARGET WORD
# ============================================================

def find_target(sentence, target=""):
    ws = words(sentence)

    # 1. Respect a target explicitly entered by the user.
    if target and target.lower() in ws:
        return target.lower()

    # 2. Find words in the sentence that have multiple
    #    distinct senses in the local WSD dataset.
    dataset_senses = {}

    for row in wsd_data:
        word = row.get("target_word", "").lower().strip()
        sense = (
            row.get("sense", "").strip()
            or row.get("meaning", "").strip()
        )

        if word in ws and word not in STOP and sense:
            dataset_senses.setdefault(word, {}).setdefault(
                sense, []
            ).append(row)

    # 3. Select a dataset word only when its context
    #    supports one sense more strongly than the others.
    candidates = []

    for word, senses in dataset_senses.items():
        if len(senses) < 2:
            continue

        context = clean_context(sentence, word)
        scores = []

        for sense, examples in senses.items():
            best_score = 0

            for row in examples:
                example_words = clean_context(
                    row.get("sentence", ""), word
                )
                score = len(context & example_words)
                best_score = max(best_score, score)

            scores.append((best_score, sense))

        scores.sort(reverse=True)
        best_score = scores[0][0]
        second_score = scores[1][0]

        if best_score >= 1 and best_score > second_score:
            candidates.append(
                (best_score - second_score, best_score, word)
            )

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][2]

    # 4. WordNet fallback: require contextual evidence
    #    favoring one sense over another.
    try:
        wordnet_candidates = []

        for word in ws:
            if word in STOP or len(word) < 3:
                continue

            synsets = wn.synsets(word)

            if len(synsets) < 2:
                continue

            context = clean_context(sentence, word)
            scores = []

            for syn in synsets:
                sense_text = (
                    syn.definition() + " "
                    + " ".join(syn.examples()) + " "
                    + " ".join(
                        lemma.name().replace("_", " ")
                        for lemma in syn.lemmas()
                    )
                )

                sense_words = clean_context(sense_text, word)
                scores.append(len(context & sense_words))

            scores.sort(reverse=True)

            if (
                scores[0] >= 1
                and scores[0] > scores[1]
            ):
                wordnet_candidates.append(
                    (scores[0] - scores[1], scores[0], word)
                )

        if wordnet_candidates:
            wordnet_candidates.sort(reverse=True)
            return wordnet_candidates[0][2]

    except LookupError:
        pass

    # 5. Do not force a target when no suitable word is found.
    return ""


# ============================================================
# WORD SENSE DISAMBIGUATION
# ============================================================

def word_sense(sentence, target=""):

    target = find_target(sentence, target)

    if not target:
        return {
            "word": "",
            "meaning": "No ambiguous word found.",
            "sense": "Unknown",
            "example": sentence
        }

    # ---------------- Dataset prediction ----------------

    examples = [
        row for row in wsd_data
        if row.get("target_word", "").lower() == target
    ]

    if examples:

        best = max(
            examples,
            key=lambda row: similarity(
                sentence,
                row["sentence"],
                target
            )
        )

        return {
            "word": target,
            "meaning": best["meaning"],
            "sense": best["sense"],
            "example": best["sentence"]
        }

    # ---------------- WordNet / Lesk fallback ----------------

    try:
        syn = lesk(
            words(sentence),
            target
        )

        if not syn:
            synsets = wn.synsets(target)
            syn = synsets[0] if synsets else None

        if syn:
            return {
                "word": target,
                "meaning": syn.definition(),
                "sense": syn.name(),
                "example": (
                    syn.examples()[0]
                    if syn.examples()
                    else sentence
                )
            }

    except LookupError:
        return {
            "word": target,
            "meaning": "WordNet data is not installed.",
            "sense": "Unavailable",
            "example": sentence
        }

    return {
        "word": target,
        "meaning": "Unable to determine the meaning.",
        "sense": "Unknown",
        "example": sentence
    }


# ============================================================
# SEMANTIC ROLE ANALYSIS
# ============================================================

def semantic_roles(sentence):
    tokens = word_tokenize(sentence)

    verbs = {
        "hit", "run", "runs", "ran", "running",
        "fly", "flies", "flew", "flying",
        "walk", "walks", "walked", "walking",
        "dance", "dances", "danced", "dancing",
        "speak", "speaks", "spoke", "speaking",
        "eat", "eats", "ate", "eating",
        "drink", "drinks", "drank", "drinking",
        "play", "plays", "played", "playing",
        "jump", "jumps", "jumped", "jumping",
        "see", "sees", "saw", "seeing",
        "write", "writes", "wrote", "writing",
        "read", "reads", "reading",
        "go", "goes", "went", "going",
        "come", "comes", "came", "coming",
        "sleep", "sleeps", "slept", "sleeping",
        "swim", "swims", "swam", "swimming",
        "talk", "talks", "talked", "talking",
        "look", "looks", "looked", "looking",
        "love", "loves", "loved",
        "like", "likes", "liked",
        "make", "makes", "made", "making",
        "take", "takes", "took", "taking",
        "is", "are", "was", "were", "am"
        "be", "been", "being"
    }

    verb = -1

    for i, word in enumerate(tokens):
        if word.lower() in verbs:
            verb = i
            break

    if verb == -1:
        return {
            "subject": "Not detected",
            "action": "Not detected",
            "object": "Not detected"
        }

    subject = " ".join(tokens[:verb]).strip()
    action = tokens[verb]
    obj = " ".join(tokens[verb + 1:]).strip()

    return {
        "subject": subject or "Not detected",
        "action": action,
        "object": obj or "Not detected"
    }
# ============================================================
# N-GRAM
# ============================================================

TRAINING = """
the cat is sitting on the mat
the dog is running in the park
the student is reading a book
the teacher is explaining the lesson
the weather is very pleasant
the sun is shining brightly
the computer is connected to the internet
the user enters a sentence
the system analyzes the sentence
the model predicts the next word
natural language processing is useful
machine learning is used in applications
the project uses python and flask
the application analyzes natural language
the student writes a program
the program produces the result
the user clicks the analyze button
the system displays the result
context helps identify the correct meaning
the word has multiple meanings
the player hit the ball with a bat 
the pllayer used a bat to hit the ball
the bat flew out of the cave 
the bat flew over the trees
the stone fell on the ground
the stone was lying near the river
the student  picked up the stone 
the boy threw the stone
the bird flew over the trees
the dog ran across the road
the girl opened the door
the boy entered the room
the student opened the book
the teacher explained the topic
the user entered the data
the system generated the result
"""

bigram = Counter()

for sentence in TRAINING.strip().split("\n"):
    ws = words(sentence)

    for i in range(len(ws) - 1):
        bigram[(ws[i], ws[i + 1])] += 1


def predict(sentence):

    ws = words(sentence)

    if not ws:
        return ["Please enter a sentence."]

    last = ws[-1]

    result = [
        (next_word, count)
        for (word, next_word), count in bigram.items()
        if word == last
    ]

    result.sort(key=lambda x: x[1], reverse=True)
    if result:
      return [word for word, count in result[:5]]

    
    return["the", "is", "and", "to", "a"]
    


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    try:
        data = request.get_json()

        sentence = data.get("sentence", "").strip()
        target = data.get("target", "").strip()
        module = data.get("module", "sense")

        if not sentence:
            return jsonify({
                "success": False,
                "message": "Please enter a sentence."
            })

        # Word Sense
        if module == "sense":

            result = word_sense(sentence, target)

            return jsonify({
                "success": True,
                "word": result["word"],
                "meaning": result["meaning"],
                "sense": result["sense"],
                "example": result["example"]
            })

        # Semantic Roles
        elif module == "semantic":

            result = semantic_roles(sentence)

            return jsonify({
                "success": True,
                "subject": result["subject"],
                "action": result["action"],
                "object": result["object"]
            })

        # Next Word Prediction
        elif module == "ngram":

            return jsonify({
                "success": True,
                "predictions": predict(sentence)
            })

        return jsonify({
            "success": False,
            "message": "Invalid module."
        })

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "success": False,
            "message": str(e)
        })


# ============================================================
# FIND FREE PORT
# ============================================================

def free_port():

    for port in range(5001, 5101):
        try:
            s = socket.socket()
            s.bind(("127.0.0.1", port))
            s.close()
            return port
        except OSError:
            pass

    return 5001


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    port = free_port()

    print("\nContextLex is running!")
    print(f"Dataset: {len(wsd_data)} sentences")
    print(f"URL: http://127.0.0.1:{port}\n")

    app.run(
        host="127.0.0.1",
        port=port,
        debug=False
    )
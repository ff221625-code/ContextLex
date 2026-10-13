ContextLex: Real-Time Context-Aware NLP Analysis and Language Prediction System

ContextLex is a web-based Natural Language Processing (NLP) project designed to analyze sentences, identify the meaning of words based on context, extract basic semantic roles, and predict the next word using language modeling techniques.
-----
Features

1. Word Sense Disambiguation

Identifies the intended meaning of an ambiguous word by analyzing its surrounding context. For example, the word crane can refer to a bird or a machine used to lift heavy objects.

2. Semantic Role Analysis

Analyzes sentence structure to identify:

- Subject/Agent: The person, object, or entity performing or being described in an action.
- Action/Predicate: The verb or action in the sentence.
- Object/Argument: The object, complement, or remaining phrase associated with the action.

3. Next Word Prediction

Uses an N-gram-based language modeling approach to suggest possible next words based on the preceding text.

Technologies Used

- Python – Core programming language
- Flask – Web application framework
- NLTK – Natural Language Processing toolkit
- HTML5 – Web page structure
- CSS3 – Styling and animated gradient interface
- JavaScript – Frontend interactions and communication with the Flask backend
--------
Project Structure

ContextLex/
├── app.py
├── requirements.txt
├── .gitignore
├── templates/
│   └── index.html
└── static/
    ├── script.js
    └── style.css

Installation and Setup

Prerequisites

- Python installed on your system
- Git (optional, for cloning the repository)

- 1. Clone the repository

git clone https://github.com/ff221625-code/ContextLex.git
cd ContextLex

- 2. Create a virtual environment

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

- 3. Install dependencies

pip install -r requirements.txt

If required, download the WordNet resources used by NLTK:

python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

- 4. Run the application

python app.py

Open the local address displayed by Flask in your browser. If the application uses port 5001, visit:

"http://127.0.0.1:5001"

How to Use

1. Open the ContextLex web application.
2. Enter a sentence in the input field.
3. Optionally specify a target word for Word Sense Disambiguation.
4. Select Analyze Sentence.
5. Review the word sense or semantic role results.
6. Use the Next Word feature to explore word prediction.
------
Example

Input sentence:

"The crane lifted the heavy container."

Expected word sense:

The word crane refers to a machine used for lifting heavy objects.

The exact output depends on the implemented analysis logic and the resources available in the environment.
-----
Project Objectives

- Explore practical applications of Natural Language Processing.
- Demonstrate context-based word meaning identification.
- Perform basic semantic role analysis.
- Implement N-gram-based next-word prediction.
- Integrate NLP techniques into an interactive Flask web application.
------
Limitations

The accuracy of the analysis depends on the implemented algorithms, predefined verb forms, training data, and available NLP resources. Semantic role analysis is a basic implementation and may not correctly interpret every sentence structure.
-------
Future Enhancements

- Improve grammatical and semantic analysis using advanced NLP models.
- Expand the language prediction training corpus.
- Improve support for complex sentence structures.
- Add multilingual text analysis.
- Enhance prediction accuracy and user experience.
-------
Author

ContextLex Project

A student project demonstrating context-aware NLP analysis and language prediction using Python and Flask.
--------
License

This project is available for educational and learning purposes. Add a formal open-source license if you decide to distribute it under specific reuse terms.


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

questions = [
    "What is artificial intelligence?",
    "What is AI?",
    "Explain artificial intelligence",
    "What is machine learning?",
    "What is ML?",
    "What is Python?",
    "Explain Python programming",
    "What is deep learning?",
    "What is data science?",
    "What is NLP?"
]

answers = [
    "Artificial Intelligence enables computers to perform tasks that normally require human intelligence.",
    "Artificial Intelligence enables computers to perform tasks that normally require human intelligence.",
    "Artificial Intelligence enables computers to perform tasks that normally require human intelligence.",
    "Machine Learning helps computers learn patterns from data.",
    "Machine Learning helps computers learn patterns from data.",
    "Python is a popular programming language used in AI and software development.",
    "Python is a popular programming language used in AI and software development.",
    "Deep Learning uses neural networks with multiple layers.",
    "Data Science involves analyzing data to extract useful information.",
    "Natural Language Processing helps computers understand and process human language."
]

# Keywords that identify each supported topic
topic_keywords = {
    "ai": ["ai", "artificial intelligence"],
    "machine learning": ["machine learning", "ml"],
    "python": ["python"],
    "deep learning": ["deep learning"],
    "data science": ["data science"],
    "nlp": ["nlp", "natural language processing"]
}

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)


def chatbot(user_question):
    text = user_question.lower().strip()

    # Identify whether the question contains a supported topic
    matched_topic = None

    for topic, keywords in topic_keywords.items():
        for keyword in keywords:
            if keyword in text:
                matched_topic = topic
                break
        if matched_topic:
            break

    # Reject questions about unsupported topics
    if matched_topic is None:
        return "Sorry, I don't know the answer to that question yet."

    # Find the most similar FAQ question
    user_vector = vectorizer.transform([text])
    scores = cosine_similarity(user_vector, question_vectors)[0]
    best_match = scores.argmax()

    # Require a reasonable similarity score
    if scores[best_match] < 0.15:
        return "Sorry, I couldn't understand your question. Please rephrase it."

    # Make sure the selected answer belongs to the detected topic
    answer_text = answers[best_match].lower()

    if matched_topic == "ai":
        valid = "artificial intelligence" in answer_text
    elif matched_topic == "machine learning":
        valid = "machine learning" in answer_text
    elif matched_topic == "python":
        valid = "python" in answer_text
    elif matched_topic == "deep learning":
        valid = "deep learning" in answer_text
    elif matched_topic == "data science":
        valid = "data science" in answer_text
    else:
        valid = "natural language processing" in answer_text

    if not valid:
        return "Sorry, I couldn't find a suitable answer."

    return answers[best_match]


print("Welcome to CodeAlpha AI FAQ Chatbot!")
print("Ask questions about AI, ML, Python, Deep Learning, Data Science or NLP.")
print("Type 'exit' to stop.")

while True:
    user_question = input("\nAsk a question: ").strip()

    if user_question.lower() == "exit":
        print("Thank you for using the chatbot!")
        break

    if not user_question:
        print("Please enter a question.")
        continue

    print("Chatbot:", chatbot(user_question))

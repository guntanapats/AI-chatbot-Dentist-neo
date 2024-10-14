from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import DataLoader
import numpy as np
from neo4j import GraphDatabase, basic_auth
import requests
import json
import faiss  # Faiss for optimized vector similarity search

# Initialize the model
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# Neo4j database credentials
URI = "neo4j://localhost"
AUTH = ("neo4j", "PASSWORD")

def run_query(query, parameters=None):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]
    driver.close()

# Cypher query to get greetings from the database
cypher_query = '''
MATCH (n:Greeting) RETURN n.name as name, n.msg_reply as reply;
'''
greeting_corpus = []
results = run_query(cypher_query)
for record in results:
    greeting_corpus.append(record['name'])
greeting_corpus = list(set(greeting_corpus))
print(greeting_corpus)

# Function to build the Faiss index for the greeting corpus
def build_faiss_index(corpus):
    # Encode the corpus to vectors using the SentenceTransformer model
    corpus_vecs = model.encode(corpus, convert_to_tensor=False, normalize_embeddings=True)
    
    # Initialize Faiss index for cosine similarity search
    dimension = corpus_vecs.shape[1]  # Size of the vectors
    index = faiss.IndexFlatIP(dimension)  # Inner Product (IP) for cosine similarity
    
    # Normalize vectors for cosine similarity (Faiss expects non-normalized vectors)
    faiss.normalize_L2(corpus_vecs)
    
    # Add the vectors to the index
    index.add(corpus_vecs)
    
    return index, corpus_vecs

# Build the Faiss index once and reuse it
faiss_index, greeting_vecs = build_faiss_index(greeting_corpus)

# Function to search for the most similar sentence using Faiss
def faiss_search(index, sentence_vec, k=1):
    # Normalize input vector for cosine similarity
    faiss.normalize_L2(sentence_vec)
    
    # Search the index for the top k nearest neighbors
    D, I = index.search(sentence_vec, k)  # D is distances, I is indices
    return D, I

# Function to perform the query to Neo4j for responses
def neo4j_search(neo_query):
    results = run_query(neo_query)
    for record in results:
        response_msg = record['reply']
    return response_msg

# Main function to compute the response
def compute_response(sentence, user_id):
    # Encode the input sentence as a vector
    ask_vec = model.encode([sentence], convert_to_tensor=False, normalize_embeddings=True)
    
    # Perform Faiss search to get the most similar greeting
    D, I = faiss_search(faiss_index, ask_vec, k=1)
    
    # Get the best match from the index
    max_greeting_score = I[0][0]
    match_greeting = greeting_corpus[max_greeting_score]
    
    if D[0][0] > 0.5:
        # If the similarity score is greater than 0.5, return the corresponding reply
        my_cypher = f"MATCH (n:Greeting) WHERE n.name = '{match_greeting}' RETURN n.msg_reply AS reply"
        my_msg = neo4j_search(my_cypher)
    else:
        # If no good match is found, fallback to Ollama API
        ollama_api_url = "http://localhost:11434/api/generate"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "model": "supachai/llama-3-typhoon-v1.5",
            "prompt": sentence + "สรุปคำตอบโดยไม่เกิน30คำ",
            "stream": False
        }
        response = requests.post(ollama_api_url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            response_data = response.text
            data = json.loads(response_data)
            my_msg = data["response"] + " คำตอบจาก Ollama"
        else:
            my_msg = f"Failed to get a response: {response.status_code}, {response.text}"

    # Save chat history to Neo4j
    save_chat_history(user_id, sentence, my_msg)
    
    return my_msg

# Function to save chat history to Neo4j
def save_chat_history(user_id, user_message, bot_response):
    query = '''
    MERGE (u:User {id: $user_id})
    CREATE (q:Question {message: $user_message, timestamp: timestamp()})
    CREATE (a:Answer {response: $bot_response, timestamp: timestamp()})
    CREATE (u)-[:ASKED]->(q)
    CREATE (q)-[:HAS_ANSWER]->(a)
    '''
    parameters = {
        'user_id': user_id,
        'user_message': user_message,
        'bot_response': bot_response
    }
    run_query(query, parameters)

# Flask app for LINE bot integration
app = Flask(__name__)

@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    try:
        json_data = json.loads(body)
        access_token = 'ACCESS_TOKEN'
        secret = 'SECRET'
        line_bot_api = LineBotApi(access_token)
        handler = WebhookHandler(secret)
        signature = request.headers['X-Line-Signature']
        handler.handle(body, signature)
        
        user_id = json_data['events'][0]['source']['userId']  # Extract userId from the received data
        msg = json_data['events'][0]['message']['text']
        tk = json_data['events'][0]['replyToken']
        response_msg = compute_response(msg, user_id)
        
        line_bot_api.reply_message(tk, TextSendMessage(text=response_msg))
        print(msg, tk)
    except Exception as e:
        print(f"Error: {e}\nBody: {body}")
    return 'OK'

if __name__ == '__main__':
    app.run(port=5000)

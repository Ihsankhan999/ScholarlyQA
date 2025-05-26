# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 03:04:58 2025

@author: TechEnclave Computer
"""

# Install required libraries
# pip install langchain-community neo4j openai transformers rouge-score bert-score google-generativeai langchain-google-genai pandas sklearn matplotlib seaborn tenacity

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from neo4j import GraphDatabase
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sklearn.metrics import precision_score, recall_score, f1_score
import re
import pandas as pd
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import google.api_core.exceptions as google_exceptions

# Step 0: Configure Gemini API
GEMINI_API_KEY = "# Replace with your Gemini API key"  
genai.configure(api_key=GEMINI_API_KEY)

# Step 1: Neo4j Retriever
class Neo4jRetriever:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def query_neo4j(self, query):
        """
        Execute a Cypher query on the Neo4j database.
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [record for record in result]

# Step 2: Query Generator
class QueryGenerator:
    def __init__(self):
        self.system_prompt = """
            You are an expert in converting natural language queries to Cypher Graph queries for a database. The Graph has the following Node Labels:Abstract ,Affiliation ,Author
    Conference,DOI,Date,Keyword,PDF_Link,Paper,Publisher, ReferenceCount. 
           
            The relationships are:
            - AUTHOR AFFILIATED_WITH Affiliation
            - Paper  HAS_DOI DOI 
            -Paper HAS_KEYWORD Keyword
            -Paper HAS_PDF_LINK PDF_Link
            - Paper Abstract HAS_ABSTRACT
            -Paper HAS_PUBLICATION_DATE  Date
            -Paper PUBLISHED_BY Publisher
            -Paper  PUBLISHED_IN Conference
            -Paper REF_BY_COUNT ReferenceCount
            -Paper WRITTEN_BY Author

            Here are some example 1 hop , 2 hop and 3 hop queries:

            Example 1 - Who is the author of paper  [Title]?
            ```
            MATCH (p:Paper {Title: '<Title>')-[:WRITTEN_BY]-(a:Author)
            RETURN a.Name;
            ```
            Example 2 - What is the DOI of [Document Title]?
            ``` 
            MATCH (p:Paper {Title: "<Title>"})-[:HAS_DOI]->(d:DOI)
            RETURN d.ID;
            ```
            Example 2 -  What is the DOI of papers written by [Author Name]?
            ``` 
           MATCH (a:Author {Name: "<Name>"})-[:WRITTEN_BY]-(p:Paper)-[:HAS_DOI]->(d:DOI)
           RETURN  d.ID;
            ```
            
            Example 3 -When was [Document Title] published?
            
            ```
            MATCH (p:Paper {Title: "<Title>"})
            RETURN p.Publication_Year;
            ```
           
            Example 4 - In which conference was [Document Title] published?
            ```
            MATCH (p:Paper {Title: "<Title>"})-[:PUBLISHED_IN]->(c:Conference)
            RETURN c.Name;

            ```
            
            Example 5 - What are the keywords associated with [Document Title]?
            MATCH (p:Paper {Title: "<Title>"})-[:HAS_KEYWORD]->(k:Keyword)
            RETURN k.Author_Keywords;
            ```
        
            Example 6. What is the citation count of [Document Title]?
            ```
           MATCH (p:Paper {Title: "Title"})-[:REF_BY_COUNT]->(r:ReferenceCount)
           RETURN r.Count;
               ```
           Example 7. Where is [Author Name] affiliated?
             ```
            MATCH (a:Author {Name: "<Name>"})-[:AFFILIATED_WITH]->(aff:Affiliation)
            RETURN aff.Institution;
            ```

          Example 8. What is the abstract of [Document Title]?
             ```
            MATCH (p:Paper {Title: "<Title>"})-[:HAS_ABSTRACT]->(abs:Abstract)
            RETURN abs.Text;
            ```
        Example 9. How many references does [Document Title] contain?
            ```
                MATCH (p:Paper {Title: "<Title>"})-[:REF_BY_COUNT]->(r:ReferenceCount)
            RETURN r.Count AS Reference_Count;
            ```

        Example 10. What is the online publication date of [Document Title]?
                ```
                MATCH (p:Paper {Title: "<Title>"})
                RETURN p.Online_Date;
                ```

        Example 11. Who is the publisher of [Conference Name]?
        ```    
        MATCH (c:Conference {Name: "<Title>"})-[:PUBLISHED_BY]->(pub:Publisher) 
        RETURN pub.Name;
        ```
        
        12. What is the PDF link of [Document Title]?
            ```
        MATCH (p:Paper {Title: "<Title>"})-[:HAS_PDF_LINK]->(pdf:PDF_Link)
        RETURN pdf.ID;
        ```
      Example 13. Which authors have contributed to [Conference Name]?
        ```
        MATCH (p:Paper)-[:PUBLISHED_IN]->(c:Conference {Name: "<Name>"}) 
        MATCH (p) -[:WRITTEN_BY]-(a:Author) 
        RETURN DISTINCT a.Name;
        ```

    // 2-Hop Queries

    Example 1. Who is the author of a paper published in [Conference Name] in [Publication Year]?

        ```
        MATCH (p:Paper)-[:PUBLISHED_IN]->(c:Conference {Name: "<Name>"})  
        WHERE p.Publication_Year = 2023  
        MATCH (p)-[:WRITTEN_BY]-(a:Author)  
        RETURN DISTINCT a.Name;
        ```

    Example2. What are the keywords of papers written by [Author Name]?
        ```
        MATCH (a:Author {Name: "<Name>"})-[:WRITTEN_BY]-(p:Paper)-[:HAS_KEYWORD]->(k:Keyword)  
        RETURN DISTINCT k.Author_Keywords;
        
        ```

    Example 3.What is the DOI of papers written by [Author Name]? 
        ```
        MATCH (a:Author {Name: "<Name>"})<-[:WRITTEN_BY]-(p:Paper)  
        RETURN p.Title, p.DOI;
        ```
    Example 4.What papers did [Author Name] publish in [Publication Year]? 
        ```
        MATCH (a:Author {Name: "<Name>"})<-[:WRITTEN_BY]-(p:Paper)  
        WHERE p.Publication_Year = 2023  
        RETURN p.Title;
        ```

    Example 5.Which papers did [Author Name] publish in [Journal Name]? 
        ```
        MATCH (p:Paper)-[:WRITTEN_BY]->(a:Author {Name: "<Name>"})
        RETURN p.Title, p.Publication_Year;
        ```

    Example 6.What are the most referenced papers in [Publication Title]? 
        ```
        MATCH (p:Paper)-[:PUBLISHED_IN]->(c:Conference {Name: "<Name>"}) 
        MATCH (p)-[:REF_BY_COUNT]->(r:ReferenceCount) 
        RETURN p.Title, r.Count AS ReferenceCount 
        ORDER BY ReferenceCount DESC 
        LIMIT 10;
        ```

    Example 7.What are the top topics covered by papers published in [Publication Year]?
        ```
        MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword)  
        WHERE p.Publication_Year = 2023  
        RETURN k.Author_Keywords, COUNT(p) AS PaperCount  
        ORDER BY PaperCount DESC  
        LIMIT 300;
        ```
    Example 8. Which papers have more than 50 references and are published in [Conference Name]?
        ```
        MATCH (p:Paper)-[:PUBLISHED_IN]->(c:Conference {Name: "<Name>"}), (p)-[:REF_BY_COUNT]->(r:ReferenceCount) 
        WHERE r.Count >10 
        RETURN p.Title;
        ```
    Example 9. What are the keywords of papers written by [Author Name]?
        ```
        MATCH (a:Author {Name: "<Name>"})-[:WRITTEN_BY]-(p:Paper)-[:HAS_KEYWORD]->(k:Keyword) 
        RETURN DISTINCT k.Author_Keywords;
        ```
    Example10 Which conferences had papers by [Author Name]?
        ```
        MATCH (a:Author {Name: "<Name>"})-[:WRITTEN_BY]-(p:Paper)-[:PUBLISHED_IN]->(c:Conference)
        RETURN DISTINCT c.Name;
        ```

    // 3-Hop Queries

    Example 1. Which institution is affiliated with the author of the paper published by [Publisher Name]?
        ```
        MATCH (p:Paper)-[:PUBLISHED_IN]->(c:Conference)-[:PUBLISHED_BY]->(pub:Publisher {Name: "<Name>"}),
              (p)-[:WRITTEN_BY]-(a:Author)-[:AFFILIATED_WITH]->(aff:Affiliation)
        RETURN DISTINCT aff.Institution;
        ```
    Example 2. Which institutions are affiliated with authors who have written papers with [Keyword]?
        ```
       
        MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword {Author_Keywords: "<Author_Keywords>"}), 
        (p)-[:WRITTEN_BY]-(a:Author)-[:AFFILIATED_WITH]->(aff:Affiliation) 
        WITH SPLIT(aff.Institution, "; ") AS institutions
        UNWIND institutions AS institution
        RETURN DISTINCT TRIM(institution) AS UniqueInstitutions;

        ```
        
    Example 3. What is the DOI of the paper written by the author affiliated with [Institution Name]?
        ```
        MATCH (a:Author)-[:AFFILIATED_WITH]->(aff:Affiliation {Institution: "<Institution>"}), 
              (a)-[:WRITTEN_BY]-(p:Paper)-[:HAS_DOI]->(d:DOI) 
        RETURN p.Title, d.ID;
        ```
    Example 4. Which keywords are associated with the paper written by the author affiliated with [Institution Name]?
        ```
        MATCH (a:Author)-[:AFFILIATED_WITH]->(aff:Affiliation {Institution: "<Institution>"}), 
              (a)-[:WRITTEN_BY]-(p:Paper)-[:HAS_KEYWORD]->(k:Keyword)
        RETURN DISTINCT k.Author_Keywords;
        ```
    Example 5.  What is the References count of the paper authored by the researcher affiliated with [Institution Name]?
        ```
        MATCH (a:Author)-[:AFFILIATED_WITH]->(aff:Affiliation {Institution: "<Institution>"}), 
              (a)-[:WRITTEN_BY]-(p:Paper)-[:REF_BY_COUNT]->(r:ReferenceCount)
        RETURN DISTINCT p.Title, r.Count;
        ```
    Example 6. Who is the author of the paper published by [Publisher Name] in [Publication Year]?

        ```
        MATCH (p:Paper {Publication_Year: 2023})-[:PUBLISHED_IN]->(c:Conference)-[:PUBLISHED_BY]->(pub:Publisher {Name: "<Name>"}), 
              (p)-[:WRITTEN_BY]-(a:Author)
        RETURN DISTINCT a.Name;
        ```
    Example 7. What is the abstract of the paper authored by the researcher affiliated with [Institution Name]?
        ```
        MATCH (a:Author)-[:AFFILIATED_WITH]->(aff:Affiliation {Institution: "Jodrey School of Computer Science, Acadia University, Wolfville, NS, Canada"}), 
              (a)-[:WRITTEN_BY]-(p:Paper)-[:HAS_ABSTRACT]->(abs:Abstract)
        RETURN DISTINCT abs.Text;
        ```
    Example 8. What are the IEEE terms of the paper published by [Publisher Name] in [Publication Year]?
        ```
        MATCH (p:Paper {Publication_Year: 2023})-[:PUBLISHED_IN]->(c:Conference)-[:PUBLISHED_BY]->(pub:Publisher {Name: "<Name>"}), 
              (p)-[:HAS_KEYWORD]->(k:Keyword)
        RETURN DISTINCT k.Author_Keywords;
        ```
    Example 9 What is the online date of the paper authored by the researcher affiliated with [Institution Name]?
        ```
        MATCH (a:Author)-[:AFFILIATED_WITH]->(aff:Affiliation {Institution: "<Institution>"}), 
              (a)-[:WRITTEN_BY]-(p:Paper)
        RETURN DISTINCT p.Title, p.Online_Date;
        ```
    Example 10. How many references are in the paper authored by the researcher affiliated with [Institution Name]?
        ```
        MATCH (a:Author)-[:AFFILIATED_WITH]->(aff:Affiliation {Institution: "<Institution>"}), 
              (a)-[:WRITTEN_BY]-(p:Paper)-[:REF_BY_COUNT]-(r:ReferenceCount)
        RETURN DISTINCT p.Title, r.Count;
        ```
    Example 11. Which conference presented the paper authored by the researcher affiliated with [Institution Name]?
        ```
        MATCH (a:Author)-[:AFFILIATED_WITH]->(aff:Affiliation {Institution: "<Institution>"}), 
              (a)-[:WRITTEN_BY]-(p:Paper)-[:PUBLISHED_IN]->(c:Conference)
        RETURN DISTINCT c.Name;
        ```
    Example 12 What is the publication title of the paper authored by the researcher affiliated with [Institution Name]?

        ```
        MATCH (a:Author)-[:AFFILIATED_WITH]->(aff:Affiliation {Institution: "<Institution>"}), 
              (a)-[:WRITTEN_BY]-(p:Paper)
        RETURN DISTINCT p.Title;
        ```
    Example 13 Which papers published by [Publisher Name] mention [Keyword]?

        ```
        MATCH (p:Paper)-[:PUBLISHED_IN]->(c:Conference)-[:PUBLISHED_BY]->(pub:Publisher {Name: "<Name>"}), 
              (p)-[:HAS_KEYWORD]->(k:Keyword {Author_Keywords: "writing instruction;computer science;curriculum;soft skills"})
        RETURN DISTINCT p.Title;
        ```
    Example 14 Which university is associated with the author of the paper published by [Publisher Name]?
        ```
        MATCH (p:Paper)-[:PUBLISHED_IN]->(c:Conference)-[:PUBLISHED_BY]->(pub:Publisher {Name: "<Name>"}), 
              (p)-[:WRITTEN_BY]-(a:Author)-[:AFFILIATED_WITH]-(aff:Affiliation)
        RETURN DISTINCT aff.Institution;
        ```
    Example 15. Which journal published the paper authored by the researcher affiliated with [Institution Name]?
        ```
        MATCH (a:Author)-[:AFFILIATED_WITH]->(aff:Affiliation {Institution: "<Institution>"}), 
              (a)-[:WRITTEN_BY]-(p:Paper)-[:PUBLISHED_IN]->(c:Conference)
        RETURN DISTINCT c.Name;
        ```
    Example 16 Which papers published in [Publication Year] mention [Keyword]?
        ```
        MATCH (p:Paper {Publication_Year: 2023})-[:HAS_KEYWORD]->(k:Keyword {Author_Keywords: "<Author_Keywords>"})
        RETURN DISTINCT p.Title;
        ```
        Example17. What is the DOI of the paper authored by the researcher affiliated with [Institution Name] in [Publication Year]?
        ```
        MATCH (a:Author)-[:AFFILIATED_WITH]->(aff:Affiliation {Institution: "<Institution>"}), 
              (a)-[:WRITTEN_BY]-(p:Paper {Publication_Year: 2023})-[:HAS_DOI]->(d:DOI)
              RETURN DISTINCT d.ID;
         ```
    Example 18.What is the PDF link of the paper authored by the researcher affiliated with [Institution Name]?
        ```
        MATCH (a:Author)-[:AFFILIATED_WITH]->(aff:Affiliation {Institution: "<Institution>"}), 
              (a)-[:WRITTEN_BY]-(p:Paper)-[:HAS_PDF_LINK]->(pdf:PDF_Link)
        RETURN DISTINCT pdf.ID;
        ```
            ...
            """
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY) 

    def generate_query(self, user_query):
        """
        Generate a Cypher query for the given user query.
        """
        system_prompt_escaped = self.system_prompt.replace("{", "{{").replace("}", "}}")
        prompt_template = ChatPromptTemplate.from_template(
            f"{system_prompt_escaped}\n\nUser Query: {user_query}\nCypher Query:"
        )
        chain = (
            {"user_query": RunnablePassthrough()}
            | prompt_template
            | self.llm
        )
        response = chain.invoke({"user_query": user_query})
        cypher_query_match = re.search(r'```(.*?)```', response.content, re.DOTALL)
        if cypher_query_match:
            # Remove extra spaces and normalize the query
            cypher_query = cypher_query_match.group(1).strip()
            cypher_query = re.sub(r'\s+', ' ', cypher_query)  # Replace multiple spaces with a single space
            
            # Trim spaces in entity values
            cypher_query = re.sub(r'\{Title:\s*"([^"]*)"\}', lambda m: f'{{Title: "{m.group(1).strip()}"}}', cypher_query)
            return cypher_query
        else:
            return "Invalid query generated."

# Step 3: LLM Response Generator
class LLMResponseGenerator:
    def __init__(self, model_name="gemini-pro"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=GEMINI_API_KEY)

    def generate_response(self, query, retrieved_data, query_type):
            """
            Generate a natural language response using an LLM and extract the relevant part.
            """
            formatted_data = "\n".join([str(record) for record in retrieved_data])
    
            # Directly return the retrieved data for specific query types
            if query_type == "conference" and len(retrieved_data) == 1:
                return retrieved_data[0].get("c.Name", "No results found in the database.")
            elif query_type == "abstract" and len(retrieved_data) == 1:
                return retrieved_data[0].get("abs.Text", "No abstract found.")
            elif query_type == "doi" and len(retrieved_data) == 1:
                return retrieved_data[0].get("d.ID", "No DOI found.")
            elif query_type == "author" and len(retrieved_data) == 1:
                return retrieved_data[0].get("a.Name", "No author found.")
    
            # Otherwise, use the LLM to generate a response
            prompt_template = ChatPromptTemplate.from_template(
                """
                Query: {query}
                Retrieved Data: {data}
                Generate a concise and direct response to the query based on the retrieved data. 
                Do not add any explanations, assumptions, or additional information.
                Use the retrieved data exactly as provided.
                """
            )
            chain = (
                {"query": RunnablePassthrough(), "data": RunnablePassthrough()}
                | prompt_template
                | self.llm
            )
            response = chain.invoke({"query": query, "data": formatted_data})
            return response.content

# Step 4: Extract Relevant Response
def extract_relevant_response(response, query_type):
    """
    Extract only the relevant part of the response based on the query type.
    """
    if query_type == "keywords":
        match = re.search(r'(?:keywords|keywords are|keywords of papers):?\s*([\w\s,;]+)', response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    elif query_type == "doi":
        match = re.search(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', response, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    elif query_type == "abstract":
        match = re.search(r'(?:abstract|abstract is|abstract of the paper):?\s*(.*?)(?:\n|\.\s|$)', response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    elif query_type == "online_date":
        match = re.search(r'\d{1,2}-[A-Za-z]{3}-\d{2,4}', response, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return response  # Return the original response if no match is found

# Step 5: Evaluation Functions
import re

def normalize_string(s):
    """
    Normalize a string by lowercasing, removing extra whitespace, and standardizing separators.
    """
    s = s.lower()
    s = re.sub(r'\s+', ' ', s).strip()  # Remove extra spaces
    s = re.sub(r'[^\w\s;,]', '', s)  # Remove punctuation except semicolons and commas
    s = re.sub(r'[;,]', ';', s)  # Replace commas with semicolons
    s = re.sub(r'\s*;\s*', ';', s)  # Remove spaces around semicolons
    return s

def exact_match(true_response, generated_response):
    """
    Calculate exact match between true response and generated response after normalization.
    """
    true_normalized = normalize_string(true_response)
    generated_normalized = normalize_string(generated_response)
    return int(true_normalized == generated_normalized)

def calculate_metrics(true_response, generated_response):
    """
    Calculate precision, recall, and F1-score.
    """
    true_tokens = set(normalize_string(true_response).split())
    generated_tokens = set(normalize_string(generated_response).split())
    precision = len(true_tokens.intersection(generated_tokens)) / len(generated_tokens) if len(generated_tokens) > 0 else 0
    recall = len(true_tokens.intersection(generated_tokens)) / len(true_tokens) if len(true_tokens) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def measure_execution_time(query_function, *args):
    """
    Measure the execution time of a function in milliseconds.
    """
    start_time = time.time()
    result = query_function(*args)
    end_time = time.time()
    return result, (end_time - start_time) * 1000  # Convert to milliseconds

def evaluate_response(true_response, generated_response, execution_time):
    """
    Evaluate the generated response using exact match, precision, recall, F1-score, and execution time.
    """
    em = exact_match(true_response, generated_response)
    precision, recall, f1 = calculate_metrics(true_response, generated_response)
    return {
        "Exact Match": em,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Execution Time (ms)": execution_time
    }
# Step 6: AI Agent Pipeline

# Step 5: AI Agent Pipeline
class AIAgentPipeline:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, llm_model="gemini-pro"):
        self.neo4j_retriever = Neo4jRetriever(neo4j_uri, neo4j_user, neo4j_password)
        self.query_generator = QueryGenerator()
        self.llm_generator = LLMResponseGenerator(llm_model)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(google_exceptions.ResourceExhausted)
    )
    def run_pipeline(self, user_query, query_type):
        """
        Run the AI agent pipeline for a given user query.
        """
        try:
            # Generate Cypher query
            cypher_query = self.query_generator.generate_query(user_query)
            print(f"Generated Cypher Query: {cypher_query}")
            if cypher_query.startswith("Invalid"):
                return cypher_query, 0

            # Execute Cypher query and measure execution time
            retrieved_data, execution_time = measure_execution_time(self.neo4j_retriever.query_neo4j, cypher_query)
            print(f"Retrieved Data: {retrieved_data}")
            if not retrieved_data:
                return "No results found in the database.", execution_time

            # Generate response using LLM
            response = self.llm_generator.generate_response(user_query, retrieved_data, query_type)
            print(f"LLM Response: {response}")
            return response, execution_time
        except google_exceptions.ResourceExhausted as e:
            print(f"ResourceExhausted error: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            return f"Error processing query: {e}", 0

# Step 6: Main Script
if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = AIAgentPipeline(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="12345678",
        llm_model="gemini-2.0-flash"
    )

    # Load queries from CSV
    query_file = "2_hop.csv"
    try:
        df = pd.read_csv(query_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(query_file, encoding='latin-1')
        except UnicodeDecodeError:
            df = pd.read_csv(query_file, encoding='Windows-1252')

    test_queries = [
        {
            "query": row["Query"],
            "true_response": row["Answer"],
            "query_type": row["Query Type"],
            "complexity": row.get("Complexity", "N/A")  # Optional column
        }
        for _, row in df.iterrows()
    ]

    # Store evaluation results
    results = []

    # Run the pipeline for each query
    for test_case in test_queries:
        user_query = test_case["query"]
        true_response = test_case["true_response"]
        query_type = test_case["query_type"]

        # Run the pipeline
        response, execution_time = pipeline.run_pipeline(user_query, query_type)
        print(f"Query: {user_query}")
        print(f"Generated Response: {response}")
        print(f"True Response: {true_response}")
        time.sleep(5) 
        # Evaluate the response
        evaluation_results = evaluate_response(true_response, response, execution_time)
        print("Evaluation Results:", evaluation_results)
        print("-" * 50)

        # Save results
        results.append({
            "Query": user_query,
            "True Response": true_response,
            "Generated Response": response,
            "Query Type": query_type,
            "Exact Match": evaluation_results["Exact Match"],
            "Precision": evaluation_results["Precision"],
            "Recall": evaluation_results["Recall"],
            "F1-Score": evaluation_results["F1-Score"],
            "Execution Time (ms)": evaluation_results["Execution Time (ms)"]
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save results to a CSV file
    output_csv = "2_hop_results.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    # Aggregate results
    aggregated_results = results_df.agg({
        "Exact Match": "mean",
        "Precision": "mean",
        "Recall": "mean",
        "F1-Score": "mean",
        "Execution Time (ms)": "mean"
    }).to_frame().T

    # Save aggregated results to a CSV file
    aggregated_output_csv = "2_hop_aggregated_results.csv"
    aggregated_results.to_csv(aggregated_output_csv, index=False)
    print(f"Aggregated results saved to {aggregated_output_csv}")

    # Visualize results
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Query Type", y="F1-Score", data=results_df, palette="viridis")
    plt.title("F1-Score by Query Type")
    plt.xlabel("Query Type")
    plt.ylabel("F1-Score")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Query Type", y="Execution Time (ms)", data=results_df, palette="magma")
    plt.title("Execution Time by Query Type")
    plt.xlabel("Query Type")
    plt.ylabel("Execution Time (ms)")
    plt.show()
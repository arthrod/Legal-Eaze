import streamlit as st
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification

# Load the Hugging Face model for contract analysis
contract_analysis_model_name = "nlpaueb/legal-bert-base-uncased"
contract_analysis = pipeline("ner", model=contract_analysis_model_name, tokenizer=contract_analysis_model_name)

# Load DistilBERT model for vulnerability detection
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
vulnerability_detection = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Load the BART model for summarization
summarization_model_name = "facebook/bart-large-cnn"
summarization = pipeline("summarization", model=summarization_model_name, tokenizer=summarization_model_name)

# Streamlit UI
def main():
    st.title("Advanced Legal Analysis App")

    # Input contract text
    contract_text = st.text_area("Paste the contract text here:")

    if st.button("Analyze Contract"):
        analyzed_entities = analyze_contract(contract_text)
        vulnerability_detected = detect_vulnerabilities(contract_text)
        summarization_result = summarize_contract(contract_text)
        
        display_analyzed_entities(analyzed_entities)
        
        if vulnerability_detected:
            st.warning("Vulnerabilities Detected!")
        else:
            st.success("No Vulnerabilities Detected.")
        
        display_summarization(summarization_result)

        

# Analyze contract using NER
def analyze_contract(contract_text):
    analyzed_entities = contract_analysis(contract_text)
    return analyzed_entities

# Detect vulnerabilities using DistilBERT
def detect_vulnerabilities(text):
    # Implement vulnerability detection logic here using the text classification model
    # For example, check for keywords or patterns indicating vulnerabilities
    
    # Truncate the text to a maximum length
    max_text_length = 512  # Choose a suitable maximum length
    truncated_text = text[:max_text_length]
    
    # Placeholder logic: Detect vulnerability if "confidential" is mentioned
    result = vulnerability_detection(truncated_text)
    predicted_label = result[0]['label']
    
    return predicted_label == "1"


# Summarize contract using BART
def summarize_contract(contract_text):
    summarization_result = summarization(contract_text, max_length=150, min_length=40, do_sample=True)
    return summarization_result[0]['summary_text']

# Display analyzed entities
def display_analyzed_entities(entities):
    st.header("Analyzed Entities:")
    
    for entity in entities:
        entity_text = entity.get('word', entity.get('entity', 'N/A'))
        entity_label = entity.get('entity_group', entity.get('label', 'N/A'))
        
        

# Display summarized contract
def display_summarization(summary):
    st.header("Summarized Contract:")
    st.write(summary)

# Display additional features

    # ... other suggested features ...

if __name__ == "__main__":
    main()

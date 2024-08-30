import streamlit as st
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import pandas as pd
import time


# Load the text data
file_path = 'radiologue.csv'
text_data = pd.read_csv(file_path)

# Initialize the OpenAI model
api_key = st.secrets["OPENAI_API_KEY"]
llm = OpenAI(api_key=api_key)

# Define the prompt template for health statistics data
prompt_template = PromptTemplate(
    input_variables=["data_description", "question"],
    template="""
    You are a radiology doctor in the Caribbean. You have access to the following health statistics data:
    {data_description}

    Question: {question}

    Please provide a detailed answer based on the data.
    """
)

# Create the LangChain
chain = LLMChain(llm=llm, prompt=prompt_template)

# Function to generate data description for the health statistics
def generate_data_description():
    sample_entries = text_data.sample(min(len(text_data), 5))  # Get up to 5 random rows
    description = "The dataset contains various health statistics over different years. It includes data points such as Infant Mortality Rate, Life Expectancy, Maternal Mortality Rate, Prevalence of Diabetes, and Prevalence of Hypertension. The data is structured with the following columns:\n"
    description += "- Keyword: A list of radiology.\n"
    description += "- Response: The definition or meaning of all the keywords\n\n"
    description += "Example entries:\n"
    description += "\n".join(f"Keyword: {row['Keyword']}, Response: {row['Response']}" for _, row in sample_entries.iterrows())
    return description

def get_response(question):
    data_description = generate_data_description()
    attempt = 0
    while True:
        try:
            response = chain.run(data_description=data_description, question=question)
            return response
        except Exception as e:
            error_message = str(e)
            if 'Rate limit' in error_message or 'quota' in error_message:
                wait_time = 2 ** attempt  # Exponential backoff
                st.write(f"Rate limit exceeded: {error_message}. Waiting for {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                attempt += 1
                if attempt > 5:  # Limit the number of retries
                    st.write("Exceeded maximum retry attempts.")
                    break
            else:
                st.write(f"An error occurred: {e}")
                break

# Streamlit UI
st.title("Radiology Data Query System")
st.logo('logo.png', width=300)
# Input for the user question
user_question = st.text_input("Please enter your question:")

if user_question:
    st.chat_message("user").write(user_question)
    answer = get_response(user_question)
    st.chat_message("assistant").write(answer)

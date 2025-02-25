import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Initialize Groq API
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]  
llm = ChatGroq(model_name="llama-3.3-70b-specdec", api_key=GROQ_API_KEY)

# Define Prompt Templates
assignment_prompt = PromptTemplate(
    input_variables=["subject", "question"],
    template="Provide a detailed answer for the {subject} question: {question}."
)

math_prompt = PromptTemplate(
    input_variables=["problem"],
    template="Solve the following math problem step by step: {problem}."
)

research_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Provide a well-researched summary on the topic: {topic}."
)

# Create Chains
assignment_chain = LLMChain(llm=llm, prompt=assignment_prompt)
math_chain = LLMChain(llm=llm, prompt=math_prompt)
research_chain = LLMChain(llm=llm, prompt=research_prompt)

# Streamlit UI
st.set_page_config(page_title="Student AI Assistant", layout="wide")
st.title("📚 AI Student Assistant")

# Assignment Help
st.header("📝 Assignment Help")
subject = st.text_input("Enter Subject:")
question = st.text_area("Enter Assignment Question:")
if st.button("Get Answer"):
    response = assignment_chain.run({"subject": subject, "question": question})
    st.write("### Answer:")
    st.write(response)

# Math Problem Solver
st.header("➗ Math Problem Solver")
problem = st.text_area("Enter Math Problem:")
if st.button("Solve Problem"):
    solution = math_chain.run({"problem": problem})
    st.write("### Solution:")
    st.write(solution)

# Research Assistance
st.header("🔎 Research Assistance")
topic = st.text_input("Enter Research Topic:")
if st.button("Get Research Summary"):
    research_summary = research_chain.run({"topic": topic})
    st.write("### Research Summary:")
    st.write(research_summary)

st.sidebar.header("🔗 AI Capabilities")
st.sidebar.write("- Assignment Assistance")
st.sidebar.write("- Math Problem Solving")
st.sidebar.write("- Research Guidance")

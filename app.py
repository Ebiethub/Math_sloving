import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool,initialize_agent 
from langchain.callbacks import StreamlitCallbackHandler
  

## Setting up the streamlit app
st.set_page_config(page_title="Text to math problem solver and Dada serch Assistant")


groq_api_key = st.sidebar.text_input(label="Groq API Key", value=" ", type="password")

if not groq_api_key:
    st.info("Please add your api key to continue")
    st.stop()

llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

## Initializing the the tools
wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="wikipedia",
    func=wikipedia_wrapper.run,
    description="A toold for searching the internet to find the various information."
)

## Initializing the Math tool

math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="calculator",
    func=math_chain.run,
    description="A tool for answering Maths related questions."
)

prompt="""
You are agent in solving mathematical problems. Logocally arrive at the solution
and display it point wise for the question below.
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

## Maths problem tool. Combining all tools
chain=LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool=Tool(
    name="Reason tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions"
)

## Initialize agent

assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    agent=AgentType.ZERO_SHOT_REACT_DESRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state.messages:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm a MATH chatbot who can anser all your maths problems"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

"""
## Function to generate the response
def generate_response(user_question):
    response=assistant_agent.invoke({'input':question})
    return response
"""
## Interaction

question=st.text_area("Enter your question:")

if st.button("Answer"):
    if question:
        with st.spinner("Generate response..."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)
            
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb])

            st.session_state.messages.append({'role':'assistant',"content":response})
            st.write('## Response')
            st.success(response)

    else:
        st.warning("Please enter the question")
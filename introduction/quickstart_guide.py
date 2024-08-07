import time
import openai  # Import openai to handle exceptions
from dotenv import find_dotenv, load_dotenv
from langchain_openai import OpenAI  # Updated import
from langchain_core.prompts import PromptTemplate  # Updated import
from langchain.chains import LLMChain
from langchain_community.agent_toolkits import load_tools  # Updated import
from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatOpenAI  # Updated import
from langchain.agents import initialize_agent, AgentType  # Corrected import

# Load environment variables
load_dotenv(find_dotenv())

def call_llm_with_retry(llm, prompt, retries=5, delay=10):
    for attempt in range(retries):
        try:
            return llm.invoke(prompt)  # Updated method to invoke
        except openai.RateLimitError:
            if attempt < retries - 1:
                print(f"Rate limit exceeded, retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Exceeded retries for rate limit error.")
                raise
        except openai.OpenAIError as e:
            print(f"OpenAI API error: {e}")
            raise

# --------------------------------------------------------------
# LLMs: Get predictions from a language model
# --------------------------------------------------------------

llm = OpenAI(model_name="gpt-3.5-turbo")  # Updated model
prompt = "Write a poem about python and ai"
try:
    print(call_llm_with_retry(llm, prompt))
except Exception as e:
    print(f"An error occurred: {e}")

# --------------------------------------------------------------
# Prompt Templates: Manage prompts for LLMs
# --------------------------------------------------------------

prompt_template = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

print(prompt_template.format(product="Smart Apps using Large Language Models (LLMs)"))  # Print the formatted prompt

# --------------------------------------------------------------
# Chains: Combine LLMs and prompts in multi-step workflows
# --------------------------------------------------------------

llm = OpenAI(model_name="gpt-3.5-turbo")  # Updated model
prompt_template = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt_template)
print(chain.run("AI Chatbots for Dental Offices"))

prompt_template = PromptTemplate(
    input_variables=["topic"],
    template="Write an email subject for this topic {topic}?",
)

chain = LLMChain(llm=llm, prompt=prompt_template)
print(chain.run("AI Session"))

# --------------------------------------------------------------
# Agents: Dynamically Call Chains Based on User Input
# --------------------------------------------------------------

llm = OpenAI(model_name="gpt-3.5-turbo")  # Updated model

tools = load_tools(["wikipedia", "llm-math"], llm=llm)

# Initialize agent with tools, language model, and agent type.
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Now let's test it out!
try:
    result = agent.run("In what year was python released and who is the original creator? Multiply the year by 3")
    print(result)
    
    result = agent.run("In what year was Tesla released and who is the original creator? Multiply the year by 3")
    print(result)
    
    result = agent.run("In what year was Tesla born? and who is the original creator? Multiply the year by 3")
    print(result)
except Exception as e:
    print(f"An error occurred while running the agent: {e}")

# --------------------------------------------------------------
# Memory: Add State to Chains and Agents
# --------------------------------------------------------------

llm = OpenAI(model_name="gpt-3.5-turbo")  # Updated model
conversation = ConversationChain(llm=llm, verbose=True)

try:
    output = conversation.predict(input="Hi there!")
    print(output)
    
    output = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
    print(output)
except Exception as e:
    print(f"An error occurred during the conversation: {e}")

# --------------------------------------------------------------
# Chatmodels: Create Conversational Agents
# --------------------------------------------------------------

chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")  # Updated model

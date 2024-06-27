{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1cdc90f0",
   "metadata": {},
   "outputs": [],
   "source": [
    "pip install langchain"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ed0228ed",
   "metadata": {},
   "outputs": [],
   "source": [
    "pip install openai"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8d9a88a2",
   "metadata": {},
   "outputs": [],
   "source": [
    "pip install llm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b22e385b",
   "metadata": {},
   "outputs": [],
   "source": [
    "pip install langchain.agents"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3cd472ac-281b-4350-9b22-c31a9c453074",
   "metadata": {},
   "outputs": [],
   "source": [
    "pip install langchain-chroma"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fa6d1bfb-b82f-432b-a541-d6217ff24833",
   "metadata": {},
   "outputs": [],
   "source": [
    "pip install chardet"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "c4822bfe",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importing libraries and modules\n",
    "\n",
    "import os\n",
    "import openai\n",
    "import streamlit as st\n",
    "from langchain.llms import OpenAI\n",
    "from langchain.memory import ConversationBufferMemory\n",
    "from langchain.chat_models import ChatOpenAI\n",
    "\n",
    "from langchain.chains import LLMChain,RetrievalQA\n",
    "from langchain.prompts.chat import (\n",
    "    ChatPromptTemplate,\n",
    "    MessagesPlaceholder,\n",
    "    SystemMessagePromptTemplate,\n",
    "    HumanMessagePromptTemplate,\n",
    ")\n",
    "\n",
    "from langchain import PromptTemplate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "7a33c94c-cda0-4f12-a2eb-89ee10da35e3",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-06-27 13:19:15.731 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /opt/anaconda3/lib/python3.11/site-packages/ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    },
    {
     "ename": "FileNotFoundError",
     "evalue": "No secrets files found. Valid paths for a secrets.toml file are: /Users/anshaya/.streamlit/secrets.toml, /Users/anshaya/Downloads/.streamlit/secrets.toml",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[55], line 6\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;66;03m## Set up the environment\u001b[39;00m\n\u001b[1;32m      2\u001b[0m \u001b[38;5;66;03m# Load secret keys\u001b[39;00m\n\u001b[1;32m      4\u001b[0m secrets \u001b[38;5;241m=\u001b[39m st\u001b[38;5;241m.\u001b[39msecrets  \u001b[38;5;66;03m# Accessing secrets (API keys) stored securely\u001b[39;00m\n\u001b[0;32m----> 6\u001b[0m openai_api_key \u001b[38;5;241m=\u001b[39m secrets[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mopenai\u001b[39m\u001b[38;5;124m\"\u001b[39m][\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mapi_key\u001b[39m\u001b[38;5;124m\"\u001b[39m]  \u001b[38;5;66;03m# Accessing OpenAI API key from secrets\u001b[39;00m\n\u001b[1;32m      7\u001b[0m os\u001b[38;5;241m.\u001b[39menviron[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mOPENAI_API_KEY\u001b[39m\u001b[38;5;124m\"\u001b[39m] \u001b[38;5;241m=\u001b[39m openai_api_key\n",
      "File \u001b[0;32m/opt/anaconda3/lib/python3.11/site-packages/streamlit/runtime/secrets.py:305\u001b[0m, in \u001b[0;36mSecrets.__getitem__\u001b[0;34m(self, key)\u001b[0m\n\u001b[1;32m    299\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124;03m\"\"\"Return the value with the given key. If no such key\u001b[39;00m\n\u001b[1;32m    300\u001b[0m \u001b[38;5;124;03mexists, raise a KeyError.\u001b[39;00m\n\u001b[1;32m    301\u001b[0m \n\u001b[1;32m    302\u001b[0m \u001b[38;5;124;03mThread-safe.\u001b[39;00m\n\u001b[1;32m    303\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m    304\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m--> 305\u001b[0m     value \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_parse(\u001b[38;5;28;01mTrue\u001b[39;00m)[key]\n\u001b[1;32m    306\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(value, Mapping):\n\u001b[1;32m    307\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m value\n",
      "File \u001b[0;32m/opt/anaconda3/lib/python3.11/site-packages/streamlit/runtime/secrets.py:214\u001b[0m, in \u001b[0;36mSecrets._parse\u001b[0;34m(self, print_exceptions)\u001b[0m\n\u001b[1;32m    212\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m print_exceptions:\n\u001b[1;32m    213\u001b[0m         st\u001b[38;5;241m.\u001b[39merror(err_msg)\n\u001b[0;32m--> 214\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mFileNotFoundError\u001b[39;00m(err_msg)\n\u001b[1;32m    216\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mlen\u001b[39m([p \u001b[38;5;28;01mfor\u001b[39;00m p \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_file_paths \u001b[38;5;28;01mif\u001b[39;00m os\u001b[38;5;241m.\u001b[39mpath\u001b[38;5;241m.\u001b[39mexists(p)]) \u001b[38;5;241m>\u001b[39m \u001b[38;5;241m1\u001b[39m:\n\u001b[1;32m    217\u001b[0m     _LOGGER\u001b[38;5;241m.\u001b[39minfo(\n\u001b[1;32m    218\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mSecrets found in multiple locations: \u001b[39m\u001b[38;5;132;01m{\u001b[39;00m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m, \u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;241m.\u001b[39mjoin(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_file_paths)\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m. \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    219\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mWhen multiple secret.toml files exist, local secrets will take precedence over global secrets.\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    220\u001b[0m     )\n",
      "\u001b[0;31mFileNotFoundError\u001b[0m: No secrets files found. Valid paths for a secrets.toml file are: /Users/anshaya/.streamlit/secrets.toml, /Users/anshaya/Downloads/.streamlit/secrets.toml"
     ]
    }
   ],
   "source": [
    "## Set up the environment\n",
    "# Load secret keys\n",
    "\n",
    "secrets = st.secrets  # Accessing secrets (API keys) stored securely\n",
    "\n",
    "openai_api_key = secrets[\"openai\"][\"api_key\"]  # Accessing OpenAI API key from secrets\n",
    "os.environ[\"OPENAI_API_KEY\"] = openai_api_key  # Setting environment variable for OpenAI API key"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "169d1287-2cd0-45a8-8e69-fae6d69d4673",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain_community.document_loaders import TextLoader\n",
    "from langchain_openai import OpenAIEmbeddings\n",
    "from langchain_text_splitters import CharacterTextSplitter\n",
    "from langchain_chroma import Chroma\n",
    "import chardet\n",
    "\n",
    "# Load the document, split it into chunks, embed each chunk and load it into the vector store.\n",
    "\n",
    "# Detect the encoding of the file\n",
    "with open('/Users/anshaya/Downloads/Examples.txt', 'rb') as f:\n",
    "    raw_data = f.read()\n",
    "    result = chardet.detect(raw_data)\n",
    "    encoding = result['encoding']\n",
    "    \n",
    "raw_documents = TextLoader('/Users/anshaya/Downloads/Examples.txt',encoding = encoding).load()\n",
    "text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)\n",
    "documents = text_splitter.split_documents(raw_documents)\n",
    "db = Chroma.from_documents(documents, OpenAIEmbeddings())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "a11d6419",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Creating the Prompt\n",
    "\n",
    "template = \"\"\"\n",
    "\n",
    "You are an expert in rubric generation for any given type of assignment. \n",
    "\n",
    "Start by greeting the user respectfully saying something on your own like \"I can help you create a detailed rubric based on your preferences, could you please tell your name?\". \n",
    "Collect the name from the user and then follow below steps:\n",
    "Verify the user input fields selected and then say something on your own like \"thank you\".\n",
    "Gather the {{detailLevel}},{{gradingStrictness}}, {{emphasisAreas}}, {{assignmentType}} and {{assignmentStyle}} information selected by the user. \n",
    "Finally  based on the gathered preferences, use the persona pattern to take the persona of the  user and generate a rubric that matches their style. \n",
    "Lastly, ask user if you want any modification or adjustments to the rubrics generated? If the user says no then end the conversation.\n",
    "\n",
    "Below is the context of how a rubric must look, use them as a reference to create detailed rubric for user.\n",
    "\n",
    "Context : {context}\n",
    "   \n",
    "Human : {question}\n",
    "\n",
    "Assistant : \n",
    "\n",
    "\n",
    "\n",
    "\"\"\"\n",
    "\n",
    "prompt = PromptTemplate(\n",
    "input_variables=[\"context\", \"question\"], template=template\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "38e90c6c-9ba7-45a8-8d48-2a4c0fd90a85",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Define a function to find similar documents based on a given query\n",
    "\n",
    "def get_similiar_docs(query, k=1, score=False):\n",
    "  if score:\n",
    "    similar_docs = db.similarity_search_with_score(query, k=k)\n",
    "  else:\n",
    "    similar_docs = db.similarity_search(query, k=k)\n",
    "  return similar_docs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "ceab3de0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Assigning the OPENAI model and Retrieval chain\n",
    "\n",
    "model_name = \"gpt-4\"\n",
    "llm = ChatOpenAI(model_name=model_name)\n",
    "\n",
    "chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever(),chain_type_kwargs={'prompt': prompt}\n",
    "                                   )\n",
    "\n",
    "# memory = ConversationBufferMemory(memory_key=\"chat_history\")\n",
    "\n",
    "# zapier = ZapierNLAWrapper()\n",
    "# toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)\n",
    "# tools = toolkit.get_tools() "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "1dfca151",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_answer(query):\n",
    "    similar_docs = get_similiar_docs(query)\n",
    "    answer = chain({\"query\":query})\n",
    "\n",
    "    return answer['result']\n",
    "\n",
    "    \n",
    "#Agent Chain\n",
    "\n",
    "# agent_chain = initialize_agent(tools, llm = chat, agent = \"zero-shot-react-description\", verbose = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "42efb6d0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "Human:  bye\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Goodbye! It was my pleasure to assist you. Don't hesitate to reach out if you need help with anything else in the future.\n",
      "bye\n"
     ]
    }
   ],
   "source": [
    "#Conversation flow\n",
    "\n",
    "import re\n",
    "\n",
    "conversation = []\n",
    "\n",
    "while True:\n",
    "    \n",
    "    query = input(\"Human: \" )\n",
    "    conversation.append('User: ' + query)\n",
    "\n",
    "    output = get_answer(query)\n",
    "    conversation.append('Bot: ' + output)\n",
    "    \n",
    "    print(output)\n",
    "    \n",
    "    if query == \"bye\":\n",
    "        print(\"bye\")\n",
    "        break    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "376fe003",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Conversation Collected\n",
    "\n",
    "def save_conversation(conversation):\n",
    "    with open('conversation.txt', 'w') as file:\n",
    "        file.write('\\n'.join(conversation))\n",
    "\n",
    "def display_conversation():\n",
    "    with open('conversation.txt', 'r') as file:\n",
    "        conversation = file.readlines()\n",
    "\n",
    "for line in conversation:\n",
    "    print(line.strip()) \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "32f03599",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3c2b65d7",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

import torch
import os
import gradio as gr
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai.foundation_models import ModelInference
from langchain_ibm import WatsonxLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from transformers import pipeline  # For Speech-to-Text

#######------------- LLM Initialization-------------#######

# IBM Watsonx LLM credentials
project_id = "skills-network"

credentials = Credentials(
                   url="https://us-south.ml.cloud.ibm.com",
                   # api_key="<YOUR_API_KEY>"  # Normally you'd put an API key here
                  )
client = APIClient(credentials)
model_id = "ibm/granite-3-3-8b-instruct"

parameters = {
    GenParams.DECODING_METHOD: "sample",
    GenParams.MAX_NEW_TOKENS: 512,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.TEMPERATURE: 0.5,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1,
}

# Initialize IBM Watsonx LLM
## Your code here ##
llm = WatsonxLLM(
    model_id=model_id,
    url=credentials.url,
    project_id=project_id,
    params=parameters,
)

#######------------- Helper Functions-------------#######

# Function to remove non-ASCII characters
def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i) < 128)

def product_assistant(ascii_transcript):
    ## Your code here ##
    return ascii_transcript


#######------------- Prompt Template and Chain-------------#######

# Define the prompt template
template = """
Generate meeting minutes and a list of tasks based on the provided context.

Context:
{context}

Meeting Minutes:
- Key points discussed
- Decisions made

Task List:
- Actionable items with assignees and deadlines
"""

prompt = ChatPromptTemplate.from_template(template)

# Define the chain
chain = (
    ## Your code here ##
    {"context": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()

)

#######------------- Speech2text and Pipeline-------------#######

# Speech-to-text pipeline
def transcript_audio(audio_file):
    pipe = pipeline(
          ## Your code here - Please use model="openai/whisper-medium" ##
        task="automatic-speech-recognition",
        model="openai/whisper-medium"

        )
    raw_transcript = pipe(audio_file, batch_size=8)["text"]
    ascii_transcript = remove_non_ascii(raw_transcript)

    adjusted_transcript = product_assistant(ascii_transcript)
    result = chain.invoke({"context": adjusted_transcript})

    # Write the result to a file for downloading
    output_file = "meeting_minutes_and_tasks.txt"
    with open(output_file, "w") as file:
        file.write(result)

    # Return the textual result and the file for download
    return result, output_file


#######------------- Gradio Interface-------------#######

audio_input = gr.Audio(sources="upload", type="filepath", label="Upload your audio file")
output_text = gr.Textbox(label="Meeting Minutes and Tasks")
download_file = gr.File(label="Download the Generated Meeting Minutes and Tasks")

iface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=[output_text, download_file],
    title="AI Meeting Assistant",
    description="Upload an audio file of a meeting. This tool will transcribe the audio, fix product-related terminology, and generate meeting minutes along with a list of tasks."
)

iface.launch(server_name="0.0.0.0", server_port=5000)

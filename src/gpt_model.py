import openai
from typing import List, Dict
import numpy as np


def get_embeddings(config: Dict, model, embedding_items: List[str]) -> np.array:
    """
    Generates embeddings for a list of items using OpenAI's embeddings API.

    Args:
        config (Dict): Configuration dictionary containing OpenAI client and model information.
        embedding_items (List[str]): List of items for which embeddings are to be generated.
    Returns:
        np.array: A numpy array containing the embeddings for the provided items.
    """

    # Create OpenAI client
    client = create_client(config, model)
    model = config["embedding"]["model"]
    
    response = client.embeddings.create(
        input=embedding_items,
        model=model)
    
    return np.array([x.embedding for x in response.data]).reshape(-1,)


def get_completion(config: Dict, model: str, messages: List[Dict]) -> Dict:
    """
    Generates a response from the bot using OpenAI's completion API based on the provided messages and configuration.

    Args:
        config (Dict): Configuration dictionary containing OpenAI client and model information.
        api_model (str): Key to specify which API model to use from the configuration.
        messages (List[Dict]): List of message dictionaries to be processed by the bot.
        tool_choice (str, optional): Tool choice parameter for the bot, with a default of "auto".

    Returns:
        Dict: A dictionary representing the bot's response, obtained from the OpenAI completion API.
    """

    # Create OpenAI client
    client = create_client(config, model)
    model = config["completion"]["model"]
  
    # Retrieve bot tools and system configurations
    system = config["completion"]["system"]
    
    # Prepare messages for the bot
    messages_copy = messages.copy()
    messages_copy.insert(0, system)
    
    # Get response from the bot
    response = client.chat.completions.create(
        model=model,
        messages=messages_copy)
    
    return response.choices[0].message


def create_client(config: Dict, model) -> openai.AzureOpenAI:
    """
    Creates an OpenAI client using provided configuration for Azure OpenAI.

    Args:
        config (Dict): Configuration dictionary containing Azure OpenAI parameters including
                       'azure_endpoint', 'api_key', and 'api_version'.        
    Returns:
        openai.AzureOpenAI: An instance of openai.AzureOpenAI initialized with Azure endpoint, 
                            API key, and API version from the configuration.
    """

    client = openai.AzureOpenAI(
        azure_endpoint=config[model]["azure_endpoint"], 
        api_key=config[model]["api_key"],  
        api_version=config[model]["api_version"])
    
    return client
    
from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult
from typing import List, Any, Dict
import requests
import yaml
import json
import pandas as pd


# This custom class is needed to access a LLM from core
class CustomLLM(BaseLLM):
    config: Dict[str, Any] = None

    # Defines which LLM we want to access and handles the interaction
    def _call(self, prompt: str, stop: List[str] = None) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        api_url = self.config["api_endpoint"]
        model = self.config["model"]

        payload = {
            "model": model,
            "prompt": prompt,
            "stop": stop,
            "stream": False,
        }

        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
        )

        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"API call failed: {response.status_code}, {response.text}")

    # Loads the config file
    def load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        self.config = config

    # Currently not used but necessary to override because of the abstract class CustomLLM
    def _llm_type(self) -> str:
        return self.config["model"]

    # Currently not used but necessary to override because of the abstract class CustomLLM
    def _generate(self, prompts: List[str], stop: List[str] = None) -> LLMResult:
        """Generates responses for a list of prompts."""
        responses = [self._call(prompt, stop) for prompt in prompts]
        return LLMResult(generations=[[{"text": response}] for response in responses])


# Helper function to load the prompt file
def load_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r") as file:
        prompt = file.read()
    return prompt


# Helper function to load the input file
def load_input(input_path: str) -> str:
    with open(input_path, "r") as file:
        input = file.read()
    return input


# Helper function to create the output file
def create_output(response: Dict[str, Any], output_path: str) -> Dict[str, Any]:
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(response, output_file, ensure_ascii=False, indent=4)


# Helper function to load the data file
def load_data(data_path: str) -> Dict[str, Any]:
    data = pd.read_csv(data_path)
    # data_string = data.to_csv(index=False, header=False) if the data is needed as a string
    return data

from openai import OpenAI
import pandas as pd

"""
@Args:
- df: The pandas dataframe that contains necessary data.
- column_name: The name of the column you want to clean, a str type variable.
- private_key: Your API key for OpenAI, a str type variable.
- model_type: The model that you want to use, a str type variable. Like 'o1-mini', 'o1-preview', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'
@Output:
- df: The original dataframe with a new column which indicates the status of the names
"""
def gpt_query(df, column_name, private_key, model_type):
    client = OpenAI(api_key=private_key)
    guidance = """Assume that you are a data scientist. Now you are dealing with patient data from a dentistry clinic located in Toronto, Canada.
    Your role is to decide if the data is empty, valid, invalid, or wrongly spelled. Now, you will be given some people's names in English. You must give me your feedback in the following fromats:
    - If you are given empty data, return "EMPTY"
    - If you believe the name is valid, return "VALID".
    - If you believe the name is invalid, return "INVALID".
    - If you believe the name is wrongly spelled, return "WRONG, the correct name should be ...". 
    For example, if someone is named "Jason", but he typed his name like "Jsson", you should return "WRONG, the correct name should be Jason".
    """
    new_col_data = []
    for entry in df[column_name]:
        prompt = []
        prompt.append({"role": "system", "content":guidance})
        prompt.append({"role": "user", "content": entry})
        feedback = client.chat.completions.create(
            model=model_type,
            messages=prompt
        )
        new_col_data.append(feedback.choices[0].message.content)
    new_col_name = column_name+"_flag"
    df[new_col_name] = new_col_data
    return df

"""
@Args:
- data: A single entry of the names in the column that you want to clean, a str type variable.
- private_key: Your API key for OpenAI, a str type variable.
- model_type: The model that you want to use, a str type variable. Like 'o1-mini', 'o1-preview', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'
@Output:
- feedback.choices[0].message.content: The status of the name that you provided.
"""
def gpt_query_one_by_one(data, private_key, model_type):
    client = OpenAI(api_key=private_key)
    guidance = """Assume that you are a data scientist. Now you are dealing with patient data from a dentistry clinic located in Toronto, Canada.
    Your role is to decide if the data is empty, valid, invalid, or wrongly spelled. Now, you will be given some people's names in English. You must give me your feedback in the following fromats:
    - If you are given empty data, return "EMPTY"
    - If you believe the name is valid, return "VALID".
    - If you believe the name is invalid, return "INVALID".
    - If you believe the name is wrongly spelled, return "WRONG, the correct name should be ...". 
    For example, if someone is named "Jason", but he typed his name like "Jsson", you should return "WRONG, the correct name should be Jason".
    """
    prompt = []
    prompt.append({"role": "system", "content":guidance})
    prompt.append({"role": "user", "content": data})
    feedback = client.chat.completions.create(
        model=model_type,
        messages=prompt
    )
    return feedback.choices[0].message.content

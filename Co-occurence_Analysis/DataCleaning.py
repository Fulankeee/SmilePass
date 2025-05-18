import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from openai import OpenAI


class PatientDataCleaner:
    def __init__(self, dataframe: pd.DataFrame, gpt_key: str, model_type: str):
        """
        Initialization
        """
        self.dataframe = dataframe
        self.gpt_key = gpt_key
        self.model_type = model_type

    def primary_postcode_cleaning(self):
        postal_code_series = self.dataframe["zip_code"]
        # Replace spaces
        self.dataframe["zip_code"] = [code.replace(' ', '') for code in postal_code_series]
        # Verify the length of postcodes
        invalid_codes = [code for code in self.dataframe["zip_code"] if len(code) != 6]
        self.dataframe = self.dataframe[~self.dataframe["zip_code"].str.replace(' ', '').isin(invalid_codes)]

    def duration_calculate(self):
        self.dataframe["first_visit"] = pd.to_datetime(self.dataframe["first_visit"])
        self.dataframe["last_visit"] = pd.to_datetime(self.dataframe["last_visit"])
        duration_data = []
        for d1,d2 in zip(self.dataframe["last_visit"], self.dataframe["first_visit"]):
            total_months = (d1.year - d2.year) * 12 + (d1.month - d2.month)
            if d1.day < d2.day:
                total_months -= 1
            duration_data.append(total_months)
        self.dataframe['visit_duration_months'] = duration_data

    def age_calculate_filter(self):
        self.dataframe["birth_date"] = pd.to_datetime(self.dataframe["birth_date"])
        self.dataframe["first_visit"] = pd.to_datetime(self.dataframe["first_visit"])
        # Calculate the age of the patient at the last visit
        self.dataframe['age'] = self.dataframe["first_visit"].dt.year - self.dataframe["birth_date"].dt.year

        # Judge if birthday incoming
        birthday_passed = (
                (self.dataframe["first_visit"].dt.month > self.dataframe["birth_date"].dt.month) |
                ((self.dataframe["first_visit"].dt.month == self.dataframe["birth_date"].dt.month) & (
                    self.dataframe["first_visit"].dt.day >= self.dataframe["birth_date"].dt.day))
        )

        # If birthday incoming, subtract the age by 1
        self.dataframe['age'] -= (~birthday_passed).astype(int)
        # Filter df so that it only contain data of patients whose age is valid
        self.dataframe = self.dataframe[self.dataframe['age'] >= 3]

    def empty_clean(self):
        self.dataframe = self.dataframe.dropna(how='any')

    def name_clean_gpt(self):
        client = OpenAI(api_key=self.gpt_key)
        guidance = """Assume that you are a data scientist. Now you are dealing with patient data from a dentistry clinic located in Toronto, Canada.
            Your role is to decide if the data is empty, valid, invalid, or wrongly spelled. Now, you will be given some people's names in English. You must give me your feedback in the following fromats:
            - If you are given empty data, return "EMPTY"
            - If you believe the name is valid, return "VALID".
            - If you believe the name is invalid, return "INVALID".
            - If you believe the name is wrongly spelled, return "WRONG, the correct name should be ...". 
            For example, if someone is named "Jason", but he typed his name like "Jsson", you should return "WRONG, the correct name should be Jason".
            """
        new_col_data = []
        for entry in self.dataframe["fullname"]:
            prompt = []
            prompt.append({"role": "system", "content": guidance})
            prompt.append({"role": "user", "content": entry})
            feedback = client.chat.completions.create(
                model=self.model_type,
                messages=prompt
            )
            new_col_data.append(feedback.choices[0].message.content)
        new_col_name = "fullname" + "_flag"
        self.dataframe[new_col_name] = new_col_data
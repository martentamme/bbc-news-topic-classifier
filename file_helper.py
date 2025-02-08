import json
import os

import pandas as pd


class FileHelper:
    _BASE_PATH = os.path.dirname(os.path.abspath(__file__))

    @classmethod
    def get_csv_as_df(cls,
                      file_name: str,
                      extension: str = "csv",
                      column_names: list[str] = None
                      ) -> pd.DataFrame:
        file_path = f"{cls._BASE_PATH}/data/{file_name}.{extension}"
        if column_names:
            df = pd.read_csv(file_path, names=column_names, header=None)
        else:
            df = pd.read_csv(file_path)

        return df

    @classmethod
    def write_model(cls, file_name: str, data: dict):
        folder_path = f"{cls._BASE_PATH}/model/{file_name}.json"

        with open(folder_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def read_model(cls, file_name: str) -> dict:
        folder_path = f"{cls._BASE_PATH}/model/{file_name}.json"

        with open(folder_path, "r", encoding="utf-8") as f:
            return json.load(f)

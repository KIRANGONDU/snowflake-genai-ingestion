import re
from .base_column_mapper import BaseColumnMapper

class MockGenAIColumnMapper(BaseColumnMapper):

    def map_columns(self, source_columns: list, canonical_schema: dict) -> dict:
        mapping = {}

        for col in source_columns:
            c = self._normalize(col)

            if re.search(r"id|emp|cust", c):
                mapping[col] = "customer_id"

            elif re.search(r"firstname", c):
                mapping[col] = "firstname"

            elif re.search(r"lastname", c):
                mapping[col] = "lastname"

            elif re.search(r"name", c):
                mapping[col] = "fullname"

            elif re.search(r"gender|sex", c):
                mapping[col] = "gender"

            elif re.search(r"dob|birth|dateofbirth", c):
                mapping[col] = "dob"

            elif re.search(r"mail", c):
                mapping[col] = "email"

            elif re.search(r"phone|mobile|contact", c):
                mapping[col] = "phone"

            elif re.search(r"city|location", c):
                mapping[col] = "city"

            elif re.search(r"country", c):
                mapping[col] = "country"

            elif re.search(r"salary|sal|monthlypay", c):
                mapping[col] = "salary"
            
           
            else:
                mapping[col] = None

        return mapping

    def _normalize(self, text: str) -> str:
        return re.sub(r"[^a-zA-Z]", "", text).lower()

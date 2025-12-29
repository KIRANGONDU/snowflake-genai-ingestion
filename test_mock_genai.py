from target_table_schema import CANONICAL_SCHEMA
from ai_column_mapping.column_mapper_factory import get_column_mapper

source_columns = [
    "id",
    "Name",
    "SEX",
    "DateOfBirth",
    "email",
    "phone",
    "city",
    "country",
    "extra_col"
]

mapper = get_column_mapper()
mapping = mapper.map_columns(source_columns, CANONICAL_SCHEMA)

print("Mock GenAI Mapping Result:")
print(mapping)

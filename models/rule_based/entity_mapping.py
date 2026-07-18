# Maps Presidio entity types to the project's 5-label DLP schema.
# Financial entities mirror FINPII_TYPES from notebooks/01_data_exploration.ipynb.

FINANCIAL_ENTITIES: set[str] = {
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_SSN",
    "US_BANK_NUMBER",
    "US_ITIN",
    "US_PASSPORT",
    "CRYPTO",
}

HEALTH_ENTITIES: set[str] = {
    "MEDICAL_LICENSE",
    "US_HEALTHCARE_NPI",
}

PII_ENTITIES: set[str] = FINANCIAL_ENTITIES | HEALTH_ENTITIES | {
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "PERSON",
    "LOCATION",
    "IP_ADDRESS",
    "URL",
    "DATE_TIME",
    "NRP",
    "US_DRIVER_LICENSE",
}

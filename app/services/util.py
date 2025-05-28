import re


def extract_numbers(text):
    # This pattern matches positive integers
    pattern = r'\d+'
    numbers = re.findall(pattern, text)
    # Convert the extracted strings to integers
    return [int(num) for num in numbers]
from molmass import ELEMENTS, Element

def first_uppercase_alpha(s: str):
    for char in s:
        if char.isalpha() and char.isupper():
            return char
    return None

def get_type_from_name(name: str) -> int:
    element: Element = ELEMENTS[first_uppercase_alpha(name)]
    return element.number

def get_mass_from_name(name: str) -> float:
    element: Element = ELEMENTS[first_uppercase_alpha(name)]
    return element.mass
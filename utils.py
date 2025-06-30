import re
from sklearn.preprocessing import MultiLabelBinarizer

def decode_labels(labels, mlb: MultiLabelBinarizer):
    """
    Convert one-hot encoded labels back to original string labels.
    """
    return mlb.inverse_transform(mlb.transform(labels))

def normalize_solidity_code(code):
    # Remove single-line comments
    code = re.sub(r'//.*', '', code)
    # Remove multi-line comments
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    # Remove extra whitespace
    code = re.sub(r'\s+', ' ', code)

    # Protect important Solidity keywords
    keywords = [
        'address', 'uint256', 'require', 'msg', 'sender', 'call', 'value',
        'function', 'public', 'private', 'external', 'internal', 'view', 'returns'
    ]
    keywords_pattern = r'\b(?:' + '|'.join(re.escape(k) for k in keywords) + r')\b'

    # Replace all variable and function names that are not keywords
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', code)
    unique_tokens = set(tokens)
    replace_map = {}
    counter = 1

    for token in unique_tokens:
        if not re.fullmatch(keywords_pattern, token):
            replace_map[token] = f'VAR{counter}'
            counter += 1

    # Replace tokens in code
    for original, replacement in replace_map.items():
        code = re.sub(rf'\b{original}\b', replacement, code)

    return code.strip()
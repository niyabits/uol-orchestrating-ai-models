import re

def extract_first_json_obj(s: str) -> str:
    s = re.sub(r"```(?:json)?|```", "", s, flags=re.IGNORECASE)
    start = s.find("{")
    if start == -1:
        raise RuntimeError("No opening brace found.")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(s[start:], start):
        if in_str:
            if esc: esc = False
            elif ch == '\\': esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    raise RuntimeError("Unbalanced braces; JSON not closed.")
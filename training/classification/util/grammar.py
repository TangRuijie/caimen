import re

pro_w = 'GRAMMAR'
op_words = []

def couple_brack(text):
    assert text[0] == '{'
    stack = [1]
    for i, e in enumerate(text[1:]):
        if e == '{':
            stack.append(1)
        else:
            stack.pop(1)
        if len(stack) == 0:
            return i + 1



def clear_eq(text):

    '''step 1: clear begin'''

    new_text = ''

    re_info = re.search("\\begin", text)
    last_end = 0
    while re_info:
        tmp_stack = [1]
        loc_span = re_info.span()
        new_text += text[last_end: loc_span[0]]
        text = text[loc_span[0]:]

        while len(tmp_stack) > 0:
            b_info = re.search("\\begin", text)
            e_info = re.search("\\end", text)

            if b_info is None or e_info.span()[0] < e_info.span()[0]:
                tmp_stack.pop()
                text = text[e_info.span()[1]:]
            else:
                tmp_stack.append(1)
                text = text[b_info.span()[1]:]

        brack_ind = couple_brack(text)
        text = text[brack_ind + 1:]


def contain_at_least_one(s, w_list):
    has_w = False
    for w in w_list:
        if w in s:
            has_w = True
            break
    return has_w

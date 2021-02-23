def fit_inside(w1, h1, w2, h2):
    fatness1 = w1 / h1
    fatness2 = w2 / h2
    if fatness2 < fatness1:
        scale_ratio = w2 / w1
    else:
        scale_ratio = h2 / h1
    w3 = int(w1 * scale_ratio)
    h3 = int(h1 * scale_ratio)
    return (w3, h3)

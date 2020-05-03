import numpy as np

FLOAT_MAX = np.finfo(np.float32).max


def get_recommendations(model, test_seq, k=20, exclude_preceding=False):
    recommendations = []
    predictions = []
    sequences = test_seq.sequences[:, :-k]
    for sequence in sequences:
        prediction = -model.predict(sequence)
        if exclude_preceding:
            prediction[sequences] = FLOAT_MAX
        predictions.append(prediction)
        recommendation = prediction.argsort()[:k]
        recommendations.append(recommendation)
    return recommendations, predictions


def append_to_dict(mapper, key, value):
    if key not in mapper:
        mapper[key] = np.array([])

    map_value = mapper.get(key)
    mapper[key] = np.append(map_value, value)
    return mapper


def append_to_dict_of_sets(mapper, key, value):
    if key not in mapper:
        mapper[key] = set()

    map_value = mapper.get(key)
    map_value.update(value)
    return mapper

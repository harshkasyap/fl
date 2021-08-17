def label_flip(data, source_label, target_label, poison_percent = 0.5):
    data = list(data)
    total_occurences = len([1 for _, label in data if label == source_label])
    poison_count = poison_percent * total_occurences

    # Poison all and keep only poisoned samples
    if poison_percent == -1:
        data=[tuple([instance, target_label]) for instance, label in data if label == source_label]
        
    else:
        label_poisoned = 0
        for index, _ in enumerate(data):
            data[index] = list(data[index])
            if data[index][1] == source_label:
                data[index][1] = target_label
                label_poisoned += 1
            data[index] = tuple(data[index])
            if label_poisoned >= poison_count:
                break

    return tuple(data)
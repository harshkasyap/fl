import copy, cv2, enum, heapq, os, sys, torch
from functools import partial
from multiprocessing import Pool, Process
from mxnet import nd as mnd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import agg, fl, sim

class Knowledge(enum.Enum):
    IN = 0
    FN = 1
    PN = 2

def fang_trmean(models, f, kn = Knowledge.PN):
    model_list = list(models.values())
    model_keys = list(models.keys())
    
    v = []
    if kn is Knowledge.PN:
        model_list = model_list[:f]

    for model in model_list:
        model_arr = sim.get_mx_net_arr(model)
        v.append(model_arr.reshape(-1, 1))

    # local model poisoning attack against Trimmed-mean
    vi_shape = v[0].shape
    v_tran = mnd.concat(*v, dim=1)
    maximum_dim = mnd.max(v_tran, axis=1).reshape(vi_shape)
    minimum_dim = mnd.min(v_tran, axis=1).reshape(vi_shape)
    direction = mnd.sign(mnd.sum(mnd.concat(*v, dim=1), axis=-1, keepdims=True))
    directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim
    # let the malicious clients (first f clients) perform the attack
    for i in range(f):
        random_12 = 1. + mnd.random.uniform(shape=vi_shape)
        v[i] = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
    
    dummy_model = model_list[0]
    _, d_shape = sim.get_net_arr(dummy_model)
    
    for index in range(f):
        models[model_keys[index]] = sim.get_arr_net(dummy_model, v[index].asnumpy(), d_shape)
    
    return models

def insert_trojan_plus(instance):
    instance = cv2.rectangle(instance, (13,26), (15,26), (2.8088), (1))
    instance = cv2.rectangle(instance, (14,25), (14,27), (2.8088), (1))
    return instance

def insert_trojan_pattern(instance):
    instance = cv2.rectangle(instance, (2,2), (2,2), (2.8088), (1))
    instance = cv2.rectangle(instance, (3,3), (3,3), (2.8088), (1))
    instance = cv2.rectangle(instance, (4,2), (4,2), (2.8088), (1))
    return instance

def insert_trojan_gap(instance):
    instance = cv2.rectangle(instance, (0,2), (1,2), (2.8088), (1))
    instance = cv2.rectangle(instance, (5,2), (6,2), (2.8088), (1))
    return instance

def insert_trojan_size(instance):
    instance = cv2.rectangle(instance, (0,2), (1,2), (2.8088), (1))
    instance = cv2.rectangle(instance, (3,2), (4,2), (2.8088), (1))
    return instance

def insert_trojan_pos(instance):
    instance = cv2.rectangle(instance, (2,2), (3,2), (2.8088), (1))
    instance = cv2.rectangle(instance, (2,4), (3,4), (2.8088), (1))
    instance = cv2.rectangle(instance, (5,2), (6,2), (2.8088), (1))
    instance = cv2.rectangle(instance, (5,4), (6,4), (2.8088), (1))
    return instance

def insert_trojan(client_data, target, func, poison_percent):
    client_data = list(client_data)
    total_occurences = len([1 for _, label in client_data])
    poison_count = poison_percent * total_occurences

    for index, (instance, label) in enumerate(client_data):
        if index >= poison_count:
            break
        client_data[index] = list(client_data[index])
        instance = instance.squeeze().numpy()

        # insert trojan type
        instance = func(instance)

        client_data[index][0] = torch.Tensor(instance).unsqueeze(0)
        client_data[index][1] = target
        
        client_data[index] = tuple(client_data[index])

    return list(client_data)

def label_flip(data, flip_labels, poison_percent = 0.5):
    data = list(data)
    
    for source_label, target_label in flip_labels.items():
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

def label_flip_next(data, flip_labels, poison_percent = 0.5):
    data = list(data)

    poison_count = poison_percent * len(data)
    if poison_percent == -1:
        poison_count = len(data) 

    label_poisoned = 0
    for index, _ in enumerate(data):
        data[index] = list(data[index])
        if data[index][1] in flip_labels.keys():
            data[index][1] = flip_labels[data[index][1]]
            label_poisoned += 1
        data[index] = tuple(data[index])
        if label_poisoned >= poison_count:
            break

    return tuple(data)

def layer_replacement_attack(model_to_attack, model_to_reference, layers):
    params1 = model_to_attack.state_dict().copy()
    params2 = model_to_reference.state_dict().copy()
    
    for layer in layers:
        params1[layer] = params2[layer]
    
    model = copy.deepcopy(model_to_attack)
    model.load_state_dict(params1, strict=False)
    return model

def lie_attack(models, n_attackers, kn = Knowledge.PN):
    model_list = list(models.values())
    model_keys = list(models.keys())
    
    dummy_model = model_list[0]
    _, d_shape = sim.get_net_arr(dummy_model)
    
    v = []
    if kn is Knowledge.PN:
        model_list = model_list[:n_attackers]
    elif kn is Knowledge.FN:
        model_list = model_list[n_attackers:]
        
    for model in model_list:
        model_arr, _ = sim.get_net_arr(model)
        v.append(model_arr)

    avg = np.array(v).mean(0)
    std = torch.std(torch.tensor(v), 0)

    z_values={3:0.69847, 5:0.7054, 8:0.71904, 10:0.72575, 20:0.73891}
    mal_update = avg + z_values[n_attackers] * std.numpy()
    
    for index in range(n_attackers):
        models[model_keys[index]] = sim.get_arr_net(dummy_model, mal_update, d_shape)
    
    return models

def model_poison_cosine_coord(b_arr, cosargs, c_arr):
    poison_percent = cosargs["poison_percent"] if "poison_percent" in cosargs else 1
    scale_dot = cosargs["scale_dot"] if "scale_dot" in cosargs else 1

    npd = c_arr - b_arr
    p_arr = copy.deepcopy(c_arr)
    
    dot_mb = scale_dot * sim.dot(p_arr, b_arr)
    norm_b = sim.norm(b_arr)
    norm_c = sim.norm(c_arr)
    norm_m = norm_c
    sim_mg = 1
    
    kwargs = {"scale_norm": cosargs["scale_norm"]} if "scale_norm" in cosargs else {}
    
    for index in heapq.nlargest(int(len(npd) * poison_percent), range(len(npd)), npd.take):
        p_arr, dot_mb, norm_m, sim_mg, updated = sim.cosine_coord_vector_adapter(b_arr, p_arr, index, dot_mb, norm_m, sim_mg, c_arr, norm_c, norm_b, **kwargs)
        
    params_changed = len(npd) - np.sum(p_arr == c_arr)

    return p_arr, params_changed

def sine_attack(base_model_update, cosine_args, epoch, models, n_attackers, kn = Knowledge.IN):
    model_list = list(models.values())
    mal_updates = []
    params_changed = 0

    b_arr, b_list = sim.get_net_arr(base_model_update)
            
    if epoch % cosine_args["scale_epoch"] == 0:
        cosine_args["scale_dot"] = cosine_args["scale_dot_factor"] + cosine_args["scale_dot"]
        cosine_args["scale_norm"] = cosine_args["scale_norm_factor"] * cosine_args["scale_norm"]

    if kn is Knowledge.IN:
        with Pool(n_attackers) as p:
            func = partial(model_poison_cosine_coord, b_arr, cosine_args)
            p_models = p.map(func, [sim.get_net_arr(model_list[attacker])[0] for attacker in range(n_attackers)])
            p.close()
            p.join()

        mal_updates = [sim.get_arr_net(base_model_update, p_arr, b_list) for p_arr, _ in p_models]
        params_changed = p_models[0][1]
    else:
        if kn is Knowledge.PN:
            model_list = model_list[:n_attackers]
        elif kn is Knowledge.FN:
            model_list = model_list[n_attackers:]

        model_re = np.array([sim.get_net_arr(model)[0] for model in model_list]).mean(0)
        p_model_arr, params_changed = model_poison_cosine_coord(b_arr, cosine_args, model_re)
        p_model = sim.get_arr_net(base_model_update, p_model_arr, b_list)

        mal_updates = [p_model for attacker in range(n_attackers)]
            
    return mal_updates, params_changed

def sota_agr_tailored_trmean(models, n_attackers, kn = Knowledge.PN, dev_type='unit_vec', agg_rule = agg.Rule.T_Mean, threshold=5.0, threshold_diff=1e-5):
    model_list = list(models.values())
    model_keys = list(models.keys())
    
    dummy_model = model_list[0]
    _, d_shape = sim.get_net_arr(dummy_model)
    
    v = []
    if kn is Knowledge.PN:
        model_list = model_list[:n_attackers]
    elif kn is Knowledge.FN:
        model_list = model_list[n_attackers:]

    for model in model_list:
        model_arr, _ = sim.get_net_arr(model)
        v.append(model_arr)
    model_re = np.array(v).mean(0)
    
    if dev_type == 'sign':
        deviation = np.sign(model_re)
    elif dev_type == 'unit_vec':
        # unit vector, dir opp to good dir
        deviation = model_re / sim.norm(model_re)
    elif dev_type == 'std':
        deviation = torch.std(v, 0)

    lamda = torch.Tensor([threshold])

    threshold_diff = threshold_diff
    prev_loss = -1
    lamda_fail = lamda
    lamda_succ = 0

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - (lamda * deviation).numpy())
        
        for index in range(n_attackers):
            models[model_keys[index]] = sim.get_arr_net(dummy_model, mal_update, d_shape)

        # Average
        avgargs = {"beta": n_attackers}
        agg_model = fl.federated_avg(models, None, agg_rule, **avgargs)
        agg_grads, _ = sim.get_net_arr(agg_model)

        loss = sim.norm(agg_grads - model_re)
        if prev_loss < loss:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2
        prev_loss = loss

    mal_update = (model_re - (lamda * deviation).numpy())
    
    for index in range(n_attackers):
        models[model_keys[index]] = sim.get_arr_net(dummy_model, mal_update, d_shape)
    
    return models

def sota_agnostic_min_max(models, n_attackers, kn = Knowledge.PN, dev_type='unit_vec'):
    model_list = list(models.values())
    model_keys = list(models.keys())
    
    dummy_model = model_list[0]
    _, d_shape = sim.get_net_arr(dummy_model)
    
    v = []
    if kn is Knowledge.PN:
        model_list = model_list[:n_attackers]
    elif kn is Knowledge.FN:
        model_list = model_list[n_attackers:]

    for model in model_list:
        model_arr, _ = sim.get_net_arr(model)
        v.append(model_arr)
    model_re = np.array(v).mean(0)
    
    if dev_type == 'sign':
        deviation = np.sign(model_re)
    elif dev_type == 'unit_vec':
        # unit vector, dir opp to good dir
        deviation = model_re / sim.norm(model_re)
    elif dev_type == 'std':
        deviation = torch.std(v, 0)

    lamda = torch.Tensor([50.0]).float()
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0
    
    distances = []
    for _v in v:
        distance = torch.norm((torch.tensor(v) - torch.tensor(_v)), dim=1) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
    
    max_distance = torch.max(distances)
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - (lamda * deviation).numpy())
        distance = torch.norm((torch.tensor(v) - torch.tensor(mal_update)), dim=1) ** 2
        max_d = torch.max(distance)
        
        if max_d <= max_distance:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - (lamda * deviation).numpy())

    for index in range(n_attackers):
        models[model_keys[index]] = sim.get_arr_net(dummy_model, mal_update, d_shape)
    
    return models

def sota_agnostic_min_sum(models, n_attackers, kn = Knowledge.PN, dev_type='unit_vec'):
    model_list = list(models.values())
    model_keys = list(models.keys())
    
    dummy_model = model_list[0]
    _, d_shape = sim.get_net_arr(dummy_model)
    
    v = []
    if kn is Knowledge.PN:
        model_list = model_list[:n_attackers]
    elif kn is Knowledge.FN:
        model_list = model_list[n_attackers:]

    for model in model_list:
        model_arr, _ = sim.get_net_arr(model)
        v.append(model_arr)
    model_re = np.array(v).mean(0)
    
    if dev_type == 'sign':
        deviation = np.sign(model_re)
    elif dev_type == 'unit_vec':
        # unit vector, dir opp to good dir
        deviation = model_re / sim.norm(model_re)
    elif dev_type == 'std':
        deviation = torch.std(v, 0)

    lamda = torch.Tensor([50.0]).float()
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0
    
    distances = []
    for _v in v:
        distance = torch.norm((torch.tensor(v) - torch.tensor(_v)), dim=1) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
    
    scores = torch.sum(distances, dim=1)
    min_score = torch.min(scores)
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - (lamda * deviation).numpy())
        distance = torch.norm((torch.tensor(v) - torch.tensor(mal_update)), dim=1) ** 2
        score = torch.sum(distance)
        
        if score <= min_score:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - (lamda * deviation).numpy())

    for index in range(n_attackers):
        models[model_keys[index]] = sim.get_arr_net(dummy_model, mal_update, d_shape)
    
    return models
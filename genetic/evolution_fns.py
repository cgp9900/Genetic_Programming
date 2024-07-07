import itertools
import random

from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids


def instantiate_model(model_name):
    if model_name == "KMeans":
        model_out = KMeans()
    elif model_name == "KMedoids":
        model_out = KMedoids()
    elif model_name == "DBSCAN":
        model_out = DBSCAN()
    else:
        model_out = HDBSCAN()
    return model_out


# Defining function to initialize the population
def init_population(model_params, param_values, num_models=15):
    population = []
    population_models = []

    # Random selection of models and associated parameters to initialize
    for i in range(num_models):
        rand1 = random.randint(0, len(model_params) - 1)
        curr_model_name = list(model_params[rand1].keys())[0]
        curr_model_params = list(model_params[rand1].values())[0]
        params_dict = {}

        # For each parameter, randomly select values from the dictionary
        for param in curr_model_params:
            rand2 = random.randint(0, len(param_values[param]) - 1)
            curr_param_value = param_values[param][rand2]
            if param == "metric_1" or param == "metric_2":
                param = "metric"
            elif param == "eps" and curr_model_name == "HDBSCAN":
                param = "cluster_selection_epsilon"
            else:
                pass
            params_dict[param] = curr_param_value
        population.append({curr_model_name: params_dict})

    # Take the population dictionary and iterate through to create model objects/instantiate
    for i in range(len(population)):
        model_instant = instantiate_model(list(population[i].keys())[0])
        model_instant.set_params(**list(population[i].values())[0])
        population_models.append(model_instant)

    return population_models


# Function to iterate models, fit, and evaluate cluster results.
def cluster_fitness(population, data):
    fitness_indices = {}
    for model in population:

        # Try to fit each model and evaluate wtih silhouette score since
        # some model fits may fail due to random nature of instantiating
        try:
            model.fit(data)
            curr_score = silhouette_score(data, model.labels_)
            fitness_indices[model] = curr_score
        except Exception:
            fitness_indices[model] = -1

    # Sort scoring to get most performant models at the top
    fitness_indices_sorted = {
        k: v
        for k, v in sorted(
            fitness_indices.items(), reverse=True, key=lambda item: item[1]
        )
    }

    return fitness_indices_sorted


# Function for the selection piece of the algorithm
def population_selection(dict_fitness, selection_param=0.7):
    next_generation = []

    # selecting only the top percentage of models based on parameter
    range_val = int(selection_param * len(dict_fitness))
    next_generation = dict(itertools.islice(dict_fitness.items(), range_val))

    return next_generation


# Crossover - If there are models of the same model type (e.g. several HDBSCAN)
# then take random parameters from one and swap it with another and add it
# to the population as a new model. Iterate through the process three times
# to generate three new models.


def crossover(dict_selected_pop, model_params, repeat=3):
    dict_selected_pop_agg = {}
    for model in dict_selected_pop.keys():
        model_name = type(model).__name__
        if model_name in dict_selected_pop_agg.keys():
            dict_selected_pop_agg[model_name].append([model])
        else:
            dict_selected_pop_agg[model_name] = [[model]]

    count = 0
    while count < repeat:
        rand_num = random.randint(0, len(model_params) - 1)
        rand_model = list(model_params[rand_num].keys())[0]
        try:
            len_check = len(dict_selected_pop_agg[rand_model])
        except Exception:
            len_check = 0
        if len_check > 1:
            rand_nums = random.sample(
                range(0, len(dict_selected_pop_agg[rand_model])), 2
            )
            model1 = dict_selected_pop_agg[rand_model][rand_nums[0]][0]
            model2 = dict_selected_pop_agg[rand_model][rand_nums[1]][0]
            half_param_length = int(len(model1.get_params()) / 2)
            model1_param_extract = [{k: v} for k, v in model1.get_params().items()][
                0:half_param_length
            ]
            model2_param_extract = [{k: v} for k, v in model2.get_params().items()][
                half_param_length::
            ]
            model_params_combine = model1_param_extract + model2_param_extract
            dict_model_params_mapping = {
                k: v
                for single_dict in model_params_combine
                for k, v in single_dict.items()
            }
            new_model = instantiate_model(rand_model).set_params(
                **dict_model_params_mapping
            )
            count += 1
            dict_selected_pop[new_model] = 0

    return dict_selected_pop


# Mutation - Select models at random, then select a hyperparameter at random,
# and mutate the parameter to another value in the possible selection list. Iterate
# through twice and generate two new models.


def mutation(dict_selected_pop, model_params, param_values, repeat=2):
    count = 0
    while count < repeat:
        curr_rand_model = random.choice(list(dict_selected_pop.keys()))
        curr_rand_model_name = type(curr_rand_model).__name__
        curr_rand_model_params = curr_rand_model.get_params()
        rand_param_switch = random.choice(list(curr_rand_model_params.keys()))
        if rand_param_switch == "metric" and (
            curr_rand_model_name == "KMedoids" or curr_rand_model_name == "DBSCAN"
        ):
            rand_param_switch = "metric_1"
        elif rand_param_switch == "metric" and curr_rand_model_name == "HDBSCAN":
            rand_param_switch == "metric_2"
        elif rand_param_switch == "cluster_selection_epsilon":
            rand_param_switch == "eps"
        else:
            pass

        try:
            rand_new_value = random.choice(param_values[rand_param_switch])
            flag_raise = 0
        except Exception:
            flag_raise = 1

        if flag_raise == 1:
            pass
        else:
            if rand_param_switch == "metric_1" or rand_param_switch == "metric_2":
                rand_param_switch = "metric"
            elif rand_param_switch == "eps" and curr_rand_model_name == "HDBSCAN":
                rand_param_switch = "cluster_selection_epsilon"
            else:
                pass
            curr_rand_model_params[rand_param_switch] = rand_new_value
            new_model = instantiate_model(curr_rand_model_name).set_params(
                **curr_rand_model_params
            )
            dict_selected_pop[new_model] = 0
            count += 1

    return dict_selected_pop


# Defining the evolution function - combines all previous functions and applies them iteratively


def evolution(
    model_params,
    param_values,
    init_population_num,
    df,
    selection_param,
    crossover_repeat,
    mutation_repeat,
    cutoff_score,
):
    init_pop = init_population(model_params, param_values, init_population_num)
    first_fitness_eval = cluster_fitness(population=init_pop, data=df)
    score = [element for element in first_fitness_eval.values()][0]
    model_hold = {}
    if score < cutoff_score:
        evaluation_i = first_fitness_eval
    else:
        return [model for model in first_fitness_eval.keys()][0], score
    i = 0
    while score < cutoff_score:
        if i < 70:
            select_i = population_selection(
                evaluation_i, selection_param=selection_param
            )
            crossover_i = crossover(
                dict_selected_pop=select_i,
                model_params=model_params,
                repeat=crossover_repeat,
            )
            mutation_i = mutation(
                dict_selected_pop=crossover_i,
                model_params=model_params,
                param_values=param_values,
                repeat=mutation_repeat,
            )
            next_population = [model_name for model_name in mutation_i.keys()]
            evaluation_i = cluster_fitness(population=next_population, data=df)
            score = [element for element in evaluation_i.values()][0]
            print(
                f"Top model: {[model for model in evaluation_i.keys()][0]}, associated score: {score}"
            )
            model_hold[[model for model in evaluation_i.keys()][0]] = score
            i += 1
        else:
            break
    return sorted(model_hold.items(), reverse=True, key=lambda item: item[1])[:10]

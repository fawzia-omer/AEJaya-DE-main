import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import time

def adaptive_rand(gen, max_gen, init_range=0.9):
    r = init_range * (1 - gen / (max_gen - 1))
    return r * np.random.rand()

def calculate_adaptive_probabilities(current_iter, max_iter):
    local_prob = 0.1 + 0.6 * (current_iter / max_iter)
    global_prob = 0.45 - 0.25 * (current_iter / max_iter)
    de_prob = 0.45 - 0.25 * (current_iter / max_iter)

    total = local_prob + global_prob + de_prob
    return local_prob/total, global_prob/total, de_prob/total

def calculate_probabilistic_score(current_iter, max_iter):
    local_prob, global_prob, de_prob = calculate_adaptive_probabilities(current_iter, max_iter)
    return np.random.choice(['local', 'global', 'de'], p=[local_prob, global_prob, de_prob])

def DE_mutation(X, i, F=0.8):
    a, b, c = np.random.choice(range(len(X)), 3, replace=False)
    while a == i or b == i or c == i:
        a, b, c = np.random.choice(range(len(X)), 3, replace=False)
    return X[a] + F * (X[b] - X[c])

def DE_crossover(target, mutant, CR=0.9):
    crossover = np.random.rand(len(target)) < CR
    return np.where(crossover, mutant, target)

def EJAYA_DE(fhd, X, nPop, nVar, VarMin, VarMax, MaxIt):
    population = VarMin + np.random.rand(nPop, nVar) * (VarMax - VarMin)
    fitness = np.array([fhd(individual) for individual in population])

    gen = 0
    BestAccuracy = np.zeros(MaxIt + 1)
    BestAccuracy[0] = 1 - np.min(fitness)
    XTarget = population[np.argmin(fitness)]
    X_old = population.copy()

    print(f"Iteration 0: Best Accuracy = {BestAccuracy[0]:.4f}")

    while gen < MaxIt:
        Best = population[np.argmin(fitness)]
        Worst = population[np.argmax(fitness)]
        M = np.mean(population, axis=0)

        r1, r2, r3, r4, r5, r6 = [adaptive_rand(gen, MaxIt) for _ in range(6)]

        Pu = r1 * Best + (1 - r1) * M
        Pl = r2 * Worst + (1 - r2) * M

        if np.random.rand() <= 0.5:
            X_old = population.copy()
        X_old = X_old[np.random.permutation(nPop)]

        new_population = np.zeros_like(population)
        for i in range(nPop):
            strategy = calculate_probabilistic_score(gen, MaxIt)
            if strategy == 'local':
                new_population[i] = population[i] + r3 * (Pu - population[i]) - r4 * (Pl - population[i])
            elif strategy == 'global':
                kappa = np.random.randn()
                new_population[i] = population[i] + kappa * (X_old[i] - population[i])
            elif strategy == 'de':
                mutant = DE_mutation(population, i)
                new_population[i] = DE_crossover(population[i], mutant)

        new_population = np.clip(new_population, VarMin, VarMax)
        new_fitness = np.array([fhd(individual) for individual in new_population])

        improved = new_fitness < fitness
        population[improved] = new_population[improved]
        fitness[improved] = new_fitness[improved]

        gen += 1
        BestAccuracy[gen] = 1 - np.min(fitness)
        XTarget = population[np.argmin(fitness)]

        print(f"Iteration {gen}: Best Accuracy = {BestAccuracy[gen]:.4f}")

    BestValue = np.min(fitness)
    return BestAccuracy, BestValue, XTarget

def objective_function(solution):
    mask = solution > 0.5
    num_selected = np.sum(mask)

    if num_selected == 0:
        return 1e10

    X_train_selected = X_train[:, mask]
    X_test_selected = X_test[:, mask]

    model = CatBoostClassifier(random_state=42, verbose=False)
    model.fit(X_train_selected, y_train)
    accuracy = model.score(X_test_selected, y_test)

    return 1 - accuracy

def main():
    global X_train, X_test, y_train, y_test

    train_data = pd.read_csv('UNSW_NB_15_preprocessed_train_data.csv')
    test_data = pd.read_csv('UNSW_NB_15_preprocessed_test_data.csv')

    X_train = train_data.drop('label', axis=1).values
    y_train = train_data['label'].values
    X_test = test_data.drop('label', axis=1).values
    y_test = test_data['label'].values

    feature_names = train_data.drop('label', axis=1).columns.tolist()

    nPop = 50
    nVar = X_train.shape[1]
    VarMin = 0
    VarMax = 1
    MaxIt = 100

    start_time = time.time()
    BestAccuracy, BestValue, XTarget = EJAYA_DE(objective_function, X_train, nPop, nVar, VarMin, VarMax, MaxIt)
    train_time = time.time() - start_time

    best_features = XTarget > 0.5
    X_train_selected = X_train[:, best_features]
    X_test_selected = X_test[:, best_features]

    selected_features = [feature_names[i] for i in range(len(best_features)) if best_features[i]]
    
    # Save selected features
    with open('selected_features.txt', 'w') as f:
        f.write('\n'.join(selected_features))

    # Save datasets with selected features
    train_data_selected = pd.DataFrame(X_train_selected, columns=selected_features)
    train_data_selected['label'] = y_train
    train_data_selected.to_csv('UNSW_NB_15_selected_features_train.csv', index=False)

    test_data_selected = pd.DataFrame(X_test_selected, columns=selected_features)
    test_data_selected['label'] = y_test
    test_data_selected.to_csv('UNSW_NB_15_selected_features_test.csv', index=False)

    print("\nFeature Selection Results:")
    print(f"Number of selected features: {sum(best_features)}")
    print(f"Selected features saved to 'selected_features.txt'")
    print("Dataset with selected features saved to 'UNSW_NB_15_selected_features_train.csv' and 'UNSW_NB_15_selected_features_test.csv'")
    print(f"Feature selection process took {train_time:.2f} seconds")

if __name__ == "__main__":
    main()

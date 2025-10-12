import numpy as np
from pomegranate.distributions import Normal
from pomegranate.hmm import DenseHMM
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
'''
def HMM(X_train, X_test, y_train, n_repeats=1):

    def build_model(p_to_node2):
        """Helper to construct and fit a DenseHMM with a given transition prob."""
        dists = [Normal() for _ in range(4)]
        edges = np.zeros((4, 4), dtype=float)

        # Transition probabilities
        edges[0, 1] = p_to_node2
        edges[0, 2] = 1.0 - p_to_node2
        edges[1, 3] = 1.0
        edges[2, 3] = 1.0
        edges[3, :] = 0.0

        starts = [1.0, 0.0, 0.0, 0.0]
        ends   = [0.0, 0.0, 0.0, 1.0]

        models = []
        for _ in range(n_repeats):
            model = DenseHMM(dists, edges=edges.tolist(),
                             starts=starts, ends=ends,
                             verbose=False, max_iter=50)
            model._initialize(X_train)
            model.fit(X_train)
            models.append(model)
        return models

    def get_posteriors(models, X):
        """Predict posterior probabilities, filtering out any NaN results."""
        posterior_list = []
        for i, m in enumerate(models):
            try:
                probs = m.predict_proba(X)
                if not np.isnan(probs).any():
                    posterior_list.append(probs)
                else:
                    print(f"[Model {i}] skipped due to NaN values.")
            except Exception as e:
                print(f"[Model {i}] failed: {e}")
        return posterior_list

    # --- First attempt ---
    p_to_node2 = 0.3
    models = build_model(p_to_node2)
    posterior_list = get_posteriors(models, X_test)

    # --- Retry if failed ---
    if len(posterior_list) == 0:
        print("Retrying with modified transition probability...")
        p_to_node2 = 0.6
        models = build_model(p_to_node2)
        posterior_list = get_posteriors(models, X_test)

    # --- If still failed, raise custom error ---
    if len(posterior_list) == 0:
        raise RuntimeError("Both attempts failed (NaNs in predictions).")

    # --- Align state labels using y_train ---
    # We'll use the *first model* for alignment
    m_ref = models[0]
    train_states = np.array(m_ref.predict(X_train))  # most likely state sequence

    # Determine which hidden state corresponds to "positive" (y_train==1)
    unique_states = np.unique(train_states)
    mean_label_per_state = {
        s: np.mean(y_train[train_states == s]) for s in unique_states
    }

    # The state with the highest mean label value is our "positive" state
    positive_state = max(mean_label_per_state, key=mean_label_per_state.get)

    print(f"Mapping: state {positive_state} → positive")

    # --- Aggregate results ---
    posterior_array = np.stack(posterior_list, axis=0)
    posterior_mean = np.nanmean(posterior_array, axis=0)
    post_middle = posterior_mean[:, 1, :]  # example middle timestep

    return models, posterior_mean[:, positive_state]
'''

def HMM(X_train, X_test, y_train, n_repeats=1):

    def build_model(p_to_node2):
        """Helper to construct and fit a DenseHMM with a given transition prob."""
        dists = [Normal() for _ in range(4)]
        edges = np.zeros((4, 4), dtype=float)

        # Transition probabilities
        edges[0, 1] = p_to_node2
        edges[0, 2] = 1.0 - p_to_node2
        edges[1, 3] = 1.0
        edges[2, 3] = 1.0
        edges[3, :] = 0.0

        starts = [1.0, 0.0, 0.0, 0.0]
        ends   = [0.0, 0.0, 0.0, 1.0]

        models = []
        for _ in range(n_repeats):
            model = DenseHMM(
                dists,
                edges=edges.tolist(),
                starts=starts,
                ends=ends,
                max_iter=50,
            )
            model._initialize(X_train)
            model.fit(X_train)
            models.append(model)
        return models

    def get_posteriors(models, X):
        """Predict posterior probabilities, filtering out any NaN results."""
        posterior_list = []
        for i, m in enumerate(models):
            try:
                probs = m.predict_proba(X)
                if not np.isnan(probs).any():
                    posterior_list.append(probs)
                else:
                    print(f"[Model {i}] skipped due to NaN values.")
            except Exception as e:
                print(f"[Model {i}] failed: {e}")
        return posterior_list

    # --- First attempt ---
    p_to_node2 = 0.3
    models = build_model(p_to_node2)
    posterior_list = get_posteriors(models, X_test)

    # --- Retry if failed ---
    if len(posterior_list) == 0:
        print("Retrying with modified transition probability...")
        p_to_node2 = 0.6
        models = build_model(p_to_node2)
        posterior_list = get_posteriors(models, X_test)

    # --- If still failed, raise custom error ---
    if len(posterior_list) == 0:
        print("Both attempts failed (NaNs in predictions).")
        raise RuntimeError("Both attempts failed (NaNs in predictions).")

    # --- Determine correct state assignment using training data ---
    ref_model = models[0]
    train_probs = ref_model.predict_proba(X_train)  # (n_train, seq_len, n_states)
    train_middle = train_probs[:, 1, :]             # take middle timestep probabilities

    # Compare each state's probability vs y_train and (1 - y_train)
    best_state = None
    best_distance = float("inf")

    for s in range(train_middle.shape[1]):
        p_state = train_middle[:, s]

        # Euclidean distances
        d1 = np.sum((p_state - y_train)**2)
        d2 = np.sum(((1 - p_state) - y_train)**2)

        # Pick the smaller of the two
        distance = min(d1, d2)
        if distance < best_distance:
            best_distance = distance
            best_state = s
            flipped = d2 < d1  # whether to flip probabilities

    print(f"Best matching state: {best_state}, flipped: {flipped}")

    # --- Aggregate results ---
    posterior_array = np.stack(posterior_list, axis=0)
    posterior_mean = np.nanmean(posterior_array, axis=0)
    post_middle = posterior_mean[:, 1, :]

    # Ensure output probabilities correspond to correct positive state
    prob_pos = post_middle[:, best_state]
    if flipped:
        prob_pos = 1 - prob_pos

    return models, prob_pos




'''




def HMM(X_train, X_test, n_repeats=1):

    def build_model(p_to_node2):
        """Helper to construct and fit a DenseHMM with a given transition prob."""
        dists = [Normal() for _ in range(4)]
        edges = np.zeros((4, 4), dtype=float)

        # Transition probabilities
        edges[0, 1] = p_to_node2
        edges[0, 2] = 1.0 - p_to_node2
        edges[1, 3] = 1.0
        edges[2, 3] = 1.0
        edges[3, :] = 0.0

        starts = [1.0, 0.0, 0.0, 0.0]
        ends   = [0.0, 0.0, 0.0, 1.0]

        models = []
        for _ in range(n_repeats):
            model = DenseHMM(dists, edges=edges.tolist(), starts=starts, ends=ends,
                             verbose=False, max_iter=50)
            model._initialize(X_train)
            model.fit(X_train)
            models.append(model)
        return models

    def get_posteriors(models):
        """Predict posterior probabilities, filtering out any NaN results."""
        posterior_list = []
        for i, m in enumerate(models):
            try:
                probs = m.predict_proba(X_test)
                if not np.isnan(probs).any():
                    posterior_list.append(probs)
                else:
                    print(f"[Model {i}] skipped due to NaN values.")
            except Exception as e:
                print(f"[Model {i}] failed: {e}")
        return posterior_list

    # --- First attempt ---
    p_to_node2 = 0.3
    models = build_model(p_to_node2)
    posterior_list = get_posteriors(models)

    # --- Retry if failed ---
    if len(posterior_list) == 0:
        print("Retrying with modified transition probability...")
        p_to_node2 = 0.6  # change it as you wish
        models = build_model(p_to_node2)
        posterior_list = get_posteriors(models)

    # --- If still failed, raise custom error ---
    if len(posterior_list) == 0:
        raise RuntimeError("Both attempts failed (NaNs in predictions).")

    # --- Aggregate results ---
    posterior_array = np.stack(posterior_list, axis=0)
    posterior_mean = np.nanmean(posterior_array, axis=0)
    post_middle = posterior_mean[:, 1, :]  # example middle timestep

    return models, post_middle
'''
def KNN(X_train, y_train, X_test, n_neighbors=15):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights='uniform', algorithm='auto')
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    return knn, y_pred


""" def HMM(X_train, X_test, n_repeats=1):

    dists = [Normal() for _ in range(4)]

    # Build edge matrix matching the DAG:
    # Row i is distribution over next-state given current state i.
    edges = np.zeros((4, 4), dtype=float)
    p_to_node2 = 0.3   # probability of going to state2 from state0
    p_to_node1 = 0.7   # probability of going to state1 from state0
    # state0 -> state1 or state2, no self-loop or direct to state3 in true graph.
    edges[0, 1] = p_to_node2
    edges[0, 2] = 1.0 - p_to_node2

    # state1 -> state3 (deterministic)
    edges[1, 3] = 1.0

    # state2 -> state3 (deterministic)
    edges[2, 3] = 1.0

    # state3 -> (end). We'll set no outgoing transitions (row of zeros). When using DenseHMM
    # the 'ends' parameter will allow the model to end in state3.
    edges[3, :] = 0.0

    starts = [1.0, 0.0, 0.0, 0.0]   # always start in state0
    ends =   [0.0, 0.0, 0.0, 1.0]   # must end in state3

    models = []
    for _ in range(n_repeats):

        models.append(DenseHMM(dists, edges=edges.tolist(), starts=starts, ends=ends, verbose=False, max_iter=50))
        models[-1]._initialize(X_train)   # runs k-means initialization for distribution parameters
        models[-1].fit(X_train)


    posterior_list = []

    for i, m in enumerate(models):
        try:
            probs = m.predict_proba(X_test)   # shape (n_test, seq_len, n_states)
            if not np.isnan(probs).any():
                posterior_list.append(probs)
            else:
                print(f"[Model {i}] skipped due to NaN values.")
        except Exception as e:
            print(f"[Model {i}] failed: {e}")

    # Stack and average safely
    if len(posterior_list) == 0:
        raise RuntimeError("No valid models produced predictions.")

    posterior_array = np.stack(posterior_list, axis=0)  # (n_models, n_test, seq_len, n_states)
    posterior_mean = np.nanmean(posterior_array, axis=0)  # averaged across models

    # Example: extract only middle time step (if sequence length = 2)
    post_middle = posterior_mean[:, 1, :]  # (n_test, n_states)

    return models, post_middle """
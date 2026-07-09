from itertools import combinations, product
from math import factorial
from numbers import Real


class Real_function:
    def __init__(self, boolean_function, S=None):
        if S is None:
            self.read(boolean_function)
        else:
            assert isinstance(boolean_function, int), (
                "To define a parity function, the first input must be its dimension"
            )
            assert boolean_function >= 0, "Dimension must be non-negative"

            self.boolean_function = None
            self.n_features = boolean_function
            self.function_type = "parity"
            self.parity_set = set()
            self.parity(S)

    def dim(self):
        return self.n_features

    def value(self, L):
        assert len(L) == self.n_features, "Input has the wrong number of features"
        assert all(value in (-1, 1) for value in L), "Input values must be -1 or 1"

        if self.function_type == "parity":
            result = 1
            for feature in self.parity_set:
                result *= L[feature - 1]
            return result

        if self.function_type == "sign_from_boolean":
            boolean_input = [0 if value == 1 else 1 for value in L]
            return (-1) ** self.boolean_function.value(boolean_input)

        if self.function_type == "sum":
            return self.left_function.value(L) + self.right_function.value(L)

        if self.function_type == "scalar_product":
            return self.scalar * self.base_function.value(L)

        assert False, "Unknown real function type"

    def mu(self, L, S):
        assert len(L) == self.n_features, "Input has the wrong number of features"
        assert all(value in (-1, 1) for value in L), "Input values must be -1 or 1"

        S = list(S)
        assert len(S) == len(set(S)), "S contains repeated features"
        assert all(1 <= feature <= self.n_features for feature in S), (
            "S contains a feature outside the valid range"
        )

        fixed_features = set(S)
        free_features = self.n_features - len(fixed_features)
        total = 0

        for L_prime in product([-1, 1], repeat=free_features):
            combined = []
            next_free_value = 0

            for feature in range(1, self.n_features + 1):
                if feature in fixed_features:
                    combined.append(L[feature - 1])
                else:
                    combined.append(L_prime[next_free_value])
                    next_free_value += 1

            total += self.value(combined)

        return total / (2 ** free_features)

    def SHAP(self, L, i):
        assert len(L) == self.n_features, "Input has the wrong number of features"
        assert all(value in (-1, 1) for value in L), "Input values must be -1 or 1"
        assert 1 <= i <= self.n_features, "Feature is outside the valid range"

        features_without_i = [
            feature for feature in range(1, self.n_features + 1) if feature != i
        ]
        shap_score = 0

        for size in range(self.n_features):
            for S_tuple in combinations(features_without_i, size):
                S = set(S_tuple)
                weight = (
                    factorial(size)
                    * factorial(self.n_features - size - 1)
                    / factorial(self.n_features)
                )
                marginal_contribution = self.mu(L, S | {i}) - self.mu(L, S)
                shap_score += weight * marginal_contribution

        return shap_score

    def Banzhaf(self, L, i):
        assert len(L) == self.n_features, "Input has the wrong number of features"
        assert all(value in (-1, 1) for value in L), "Input values must be -1 or 1"
        assert 1 <= i <= self.n_features, "Feature is outside the valid range"

        features_without_i = [
            feature for feature in range(1, self.n_features + 1) if feature != i
        ]
        banzhaf_score = 0

        for size in range(self.n_features):
            for S_tuple in combinations(features_without_i, size):
                S = set(S_tuple)
                banzhaf_score += self.mu(L, S | {i}) - self.mu(L, S)

        return banzhaf_score / (2 ** (self.n_features - 1))

    def parity(self, S):
        S = list(S)
        assert len(S) == len(set(S)), "S contains repeated features"
        assert all(1 <= feature <= self.n_features for feature in S), (
            "S contains a feature outside the valid range"
        )

        self.function_type = "parity"
        self.parity_set = set(S)

    def read(self, boolean_function):
        assert hasattr(boolean_function, "dim"), "Input must be a Boolean function"
        assert hasattr(boolean_function, "value"), "Input must be a Boolean function"

        self.boolean_function = boolean_function
        self.n_features = boolean_function.dim()
        self.function_type = "sign_from_boolean"
        self.parity_set = set()


def inner_product(f, g):
    assert hasattr(f, "dim"), "First input must be a real function"
    assert hasattr(f, "value"), "First input must be a real function"
    assert hasattr(g, "dim"), "Second input must be a real function"
    assert hasattr(g, "value"), "Second input must be a real function"
    assert f.dim() == g.dim(), "Real functions must have the same dimension"

    n_features = f.dim()
    total = 0

    for L in product([-1, 1], repeat=n_features):
        total += f.value(L) * g.value(L)

    return total / (2 ** n_features)


def sum(f, g):
    assert hasattr(f, "dim"), "First input must be a real function"
    assert hasattr(f, "value"), "First input must be a real function"
    assert hasattr(g, "dim"), "Second input must be a real function"
    assert hasattr(g, "value"), "Second input must be a real function"
    assert f.dim() == g.dim(), "Real functions must have the same dimension"

    result = Real_function.__new__(Real_function)
    result.boolean_function = None
    result.n_features = f.dim()
    result.function_type = "sum"
    result.parity_set = set()
    result.left_function = f
    result.right_function = g
    return result


def scalar_product(f, c):
    assert hasattr(f, "dim"), "Input must be a real function"
    assert hasattr(f, "value"), "Input must be a real function"
    assert isinstance(c, Real), "Scalar must be a real number"

    result = Real_function.__new__(Real_function)
    result.boolean_function = None
    result.n_features = f.dim()
    result.function_type = "scalar_product"
    result.parity_set = set()
    result.base_function = f
    result.scalar = c
    return result


def spectrum(g, L):
    assert hasattr(g, "dim"), "Input must be a real function"
    assert hasattr(g, "value"), "Input must be a real function"
    assert len(L) == g.dim(), "Input has the wrong number of features"
    assert all(value in (-1, 1) for value in L), "Input values must be -1 or 1"

    n_features = g.dim()
    result = {}

    for size in range(n_features + 1):
        for S_tuple in combinations(range(1, n_features + 1), size):
            S = set(S_tuple)
            parity_function = Real_function(n_features, S)
            result[frozenset(S)] = (
                inner_product(g, parity_function) * parity_function.value(L)
            )

    return result


def Fourier_coefficients(g):
    assert hasattr(g, "dim"), "Input must be a real function"
    assert hasattr(g, "value"), "Input must be a real function"

    n_features = g.dim()
    result = {}

    for size in range(n_features + 1):
        for S_tuple in combinations(range(1, n_features + 1), size):
            S = set(S_tuple)
            parity_function = Real_function(n_features, S)
            result[frozenset(S)] = inner_product(g, parity_function)

    return result


def _format_subset(S):
    if len(S) == 0:
        return "{}"

    return "{" + ", ".join(str(feature) for feature in sorted(S)) + "}"


def print_spectrum(g, L, show_zeros=True, decimals=16):
    """
    Prints the spectrum of g evaluated at input L.

    Input:
        g: A real function of dimension n.
        L: A list of length n with entries in {-1, 1}.
        show_zeros: If True, print zero entries. If False, omit them.
        decimals: Number of decimal places to print.

    Output:
        Prints one line for each subset S of {1, ..., n}. The printed value is
        <g, chi_S> * chi_S(L), where chi_S is the parity function for S.
    """
    assert isinstance(decimals, int), "Number of decimals must be an integer"
    assert decimals >= 0, "Number of decimals must be non-negative"

    values = spectrum(g, L)
    sorted_subsets = sorted(values.keys(), key=lambda S: (len(S), sorted(S)))

    if not show_zeros:
        sorted_subsets = [S for S in sorted_subsets if values[S] != 0]

    subset_strings = [_format_subset(S) for S in sorted_subsets]
    width = max([len(subset) for subset in subset_strings] + [0])

    print("Spectrum:")
    for S, subset_string in zip(sorted_subsets, subset_strings):
        value = 0 if abs(values[S]) < 1e-12 else values[S]
        print(f"{subset_string:<{width}} : {value:.{decimals}f}")


def print_Fourier_coefficients(g, show_zeros=True, decimals=16):
    """
    Prints the Fourier coefficients of g in the parity-function basis.

    Input:
        g: A real function of dimension n.
        show_zeros: If True, print zero coefficients. If False, omit them.
        decimals: Number of decimal places to print.

    Output:
        Prints one line for each subset S of {1, ..., n}. The printed value is
        the Fourier coefficient <g, chi_S>, where chi_S is the parity function
        for S.
    """
    assert isinstance(decimals, int), "Number of decimals must be an integer"
    assert decimals >= 0, "Number of decimals must be non-negative"

    values = Fourier_coefficients(g)
    sorted_subsets = sorted(values.keys(), key=lambda S: (len(S), sorted(S)))

    if not show_zeros:
        sorted_subsets = [S for S in sorted_subsets if values[S] != 0]

    subset_strings = [_format_subset(S) for S in sorted_subsets]
    width = max([len(subset) for subset in subset_strings] + [0])

    print("Fourier coefficients:")
    for S, subset_string in zip(sorted_subsets, subset_strings):
        value = 0 if abs(values[S]) < 1e-12 else values[S]
        print(f"{subset_string:<{width}} : {value:.{decimals}f}")


def print_SHAP_spectrum(g, L, i, show_zeros=True, decimals=16):
    """
    Prints the SHAP spectral decomposition of feature i at input L.

    Input:
        g: A real function of dimension n.
        L: A list of length n with entries in {-1, 1}.
        i: A feature index with 1 <= i <= n.
        show_zeros: If True, print zero entries. If False, omit them.
        decimals: Number of decimal places to print.

    Output:
        Prints one line for each subset S containing i. The printed value is
        spectrum(g, L)[S] / |S|. The last line prints the sum of these values,
        which is the SHAP score of feature i at L.
    """
    assert hasattr(g, "dim"), "Input must be a real function"
    assert hasattr(g, "value"), "Input must be a real function"
    assert len(L) == g.dim(), "Input has the wrong number of features"
    assert all(value in (-1, 1) for value in L), "Input values must be -1 or 1"
    assert 1 <= i <= g.dim(), "Feature is outside the valid range"
    assert isinstance(decimals, int), "Number of decimals must be an integer"
    assert decimals >= 0, "Number of decimals must be non-negative"

    spectrum_values = spectrum(g, L)
    shap_values = {}

    for S, value in spectrum_values.items():
        if i in S:
            shap_values[S] = value / len(S)

    sorted_subsets = sorted(shap_values.keys(), key=lambda S: (len(S), sorted(S)))

    if not show_zeros:
        sorted_subsets = [S for S in sorted_subsets if shap_values[S] != 0]

    subset_strings = [_format_subset(S) for S in sorted_subsets]
    width = max([len(subset) for subset in subset_strings] + [0])
    shap_score = 0

    print(f"SHAP spectrum for feature {i}:")
    for S, subset_string in zip(sorted_subsets, subset_strings):
        value = 0 if abs(shap_values[S]) < 1e-12 else shap_values[S]
        shap_score += value
        print(f"{subset_string:<{width}} : {value:.{decimals}f}")

    shap_score = 0 if abs(shap_score) < 1e-12 else shap_score
    print(f"SHAP score : {shap_score:.{decimals}f}")


def print_Banzhaf_spectrum(g, L, i, show_zeros=True, decimals=16):
    """
    Prints the Banzhaf spectral decomposition of feature i at input L.

    Input:
        g: A real function of dimension n.
        L: A list of length n with entries in {-1, 1}.
        i: A feature index with 1 <= i <= n.
        show_zeros: If True, print zero entries. If False, omit them.
        decimals: Number of decimal places to print.

    Output:
        Prints one line for each subset S containing i. The printed value is
        spectrum(g, L)[S] / 2^(|S|-1). The last line prints the sum of these
        values, which is the Banzhaf score of feature i at L.
    """
    assert hasattr(g, "dim"), "Input must be a real function"
    assert hasattr(g, "value"), "Input must be a real function"
    assert len(L) == g.dim(), "Input has the wrong number of features"
    assert all(value in (-1, 1) for value in L), "Input values must be -1 or 1"
    assert 1 <= i <= g.dim(), "Feature is outside the valid range"
    assert isinstance(decimals, int), "Number of decimals must be an integer"
    assert decimals >= 0, "Number of decimals must be non-negative"

    spectrum_values = spectrum(g, L)
    banzhaf_values = {}

    for S, value in spectrum_values.items():
        if i in S:
            banzhaf_values[S] = value / (2 ** (len(S) - 1))

    sorted_subsets = sorted(
        banzhaf_values.keys(), key=lambda S: (len(S), sorted(S))
    )

    if not show_zeros:
        sorted_subsets = [S for S in sorted_subsets if banzhaf_values[S] != 0]

    subset_strings = [_format_subset(S) for S in sorted_subsets]
    width = max([len(subset) for subset in subset_strings] + [0])
    banzhaf_score = 0

    print(f"Banzhaf spectrum for feature {i}:")
    for S, subset_string in zip(sorted_subsets, subset_strings):
        value = 0 if abs(banzhaf_values[S]) < 1e-12 else banzhaf_values[S]
        banzhaf_score += value
        print(f"{subset_string:<{width}} : {value:.{decimals}f}")

    banzhaf_score = 0 if abs(banzhaf_score) < 1e-12 else banzhaf_score
    print(f"Banzhaf score : {banzhaf_score:.{decimals}f}")


def print_SR_spectrum(g, L, S, show_zeros=True, decimals=16):
    """
    Prints the SR spectrum determined by subset S at input L.

    Input:
        g: A real function of dimension n.
        L: A list of length n with entries in {-1, 1}.
        S: A subset of {1, ..., n}.
        show_zeros: If True, print zero entries. If False, omit them.
        decimals: Number of decimal places to print.

    Output:
        Prints one line for each subset T such that T is not a subset of S.
        The printed value is spectrum(g, L)[T]. The last line prints the sum
        of these values.
    """
    assert hasattr(g, "dim"), "Input must be a real function"
    assert hasattr(g, "value"), "Input must be a real function"
    assert len(L) == g.dim(), "Input has the wrong number of features"
    assert all(value in (-1, 1) for value in L), "Input values must be -1 or 1"
    assert isinstance(decimals, int), "Number of decimals must be an integer"
    assert decimals >= 0, "Number of decimals must be non-negative"

    S = list(S)
    assert len(S) == len(set(S)), "S contains repeated features"
    assert all(1 <= feature <= g.dim() for feature in S), (
        "S contains a feature outside the valid range"
    )

    allowed_features = set(S)
    spectrum_values = spectrum(g, L)
    sr_values = {}

    for T, value in spectrum_values.items():
        if not set(T).issubset(allowed_features):
            sr_values[T] = value

    sorted_subsets = sorted(sr_values.keys(), key=lambda T: (len(T), sorted(T)))

    if not show_zeros:
        sorted_subsets = [T for T in sorted_subsets if sr_values[T] != 0]

    subset_strings = [_format_subset(T) for T in sorted_subsets]
    width = max([len(subset) for subset in subset_strings] + [0])
    total = 0

    print(f"SR spectrum for {_format_subset(allowed_features)}:")
    for T, subset_string in zip(sorted_subsets, subset_strings):
        value = 0 if abs(sr_values[T]) < 1e-12 else sr_values[T]
        total += value
        print(f"{subset_string:<{width}} : {value:.{decimals}f}")

    total = 0 if abs(total) < 1e-12 else total
    print(f"SUM : {total:.{decimals}f}")

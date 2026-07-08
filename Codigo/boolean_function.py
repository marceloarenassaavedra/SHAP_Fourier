import os
from itertools import combinations, product
from math import factorial


class Boolean_function:
    def __init__(self, file):
        self.read(file)

    def dim(self):
        return self.n_features

    def value(self, L):
        assert len(L) == self.n_features, "Input has the wrong number of features"
        return 1 if tuple(L) in self.true_inputs else 0

    def mu(self, L, S):
        assert len(L) == self.n_features, "Input has the wrong number of features"
        assert all(value in (0, 1) for value in L), "Input values must be 0 or 1"

        S = list(S)
        assert len(S) == len(set(S)), "S contains repeated features"
        assert all(1 <= feature <= self.n_features for feature in S), (
            "S contains a feature outside the valid range"
        )

        fixed_features = set(S)
        free_features = self.n_features - len(fixed_features)
        total = 0

        for L_prime in product([0, 1], repeat=free_features):
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
        assert all(value in (0, 1) for value in L), "Input values must be 0 or 1"
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
        assert all(value in (0, 1) for value in L), "Input values must be 0 or 1"
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

    def read(self, file):
        assert os.path.isfile(file), "File does not exist"

        true_inputs = set()
        n_features = None

        with open(file, "r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                row = line.strip()
                if row == "":
                    continue

                assert all(bit in "01" for bit in row), (
                    f"Line {line_number} contains a value different from 0 or 1"
                )

                if n_features is None:
                    n_features = len(row)
                else:
                    assert len(row) == n_features, (
                        f"Line {line_number} has the wrong number of features"
                    )

                true_inputs.add(tuple(int(bit) for bit in row))

        assert n_features is not None, "File does not define any input"

        self.n_features = n_features
        self.true_inputs = true_inputs

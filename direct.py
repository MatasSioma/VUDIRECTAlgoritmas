import numpy as np

class Rectangle:
    def __init__(self, center, edge_lengths, f_center, index):
        self.center = center          # Center point of the Rectangle (in normalized space)
        self.edge_lengths = edge_lengths  # Lengths of edges in each dimension
        self.f_center = f_center      # Function value at center
        self.index = index            # Identifier for tracking rectangles

    def measure(self):
        # Distance from center to vertices is half the diagonal length
        return 0.5 * np.linalg.norm(self.edge_lengths)

def normalize_bounds(bounds):
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    length = upper - lower
    return lower, length

def denormalize_point(point, lower, length):
    return lower + point * length

def shubert(x):
    x1, x2 = x[0], x[1]
    j = np.arange(1, 6)
    sum1 = np.sum(j * np.cos((j + 1) * x1 + j))
    sum2 = np.sum(j * np.cos((j + 1) * x2 + j))
    return sum1 * sum2

def scaled_shubert(x_scaled):
    # Scale from [0,1] to [-10,10]
    x = 20 * x_scaled - 10
    return shubert(x)

def find_potentially_optimal_rectangles(rectangles, f_min, eps):
    if not rectangles:
        return []

    d = np.array([r.measure() for r in rectangles])
    f_vals = np.array([r.f_center for r in rectangles])

    potential = []

    for j, rj in enumerate(rectangles):
        dj = d[j]
        fj = f_vals[j]

        I1 = [i for i in range(len(rectangles)) if d[i] < dj]
        I2 = [i for i in range(len(rectangles)) if d[i] > dj]
        I3 = [i for i in range(len(rectangles)) if d[i] == dj]

        # Check condition 1: fj <= f_i for all i in I3
        if not all(fj <= f_vals[i] for i in I3):
            continue

        # Calculate slopes for inequality (7)
        left_slope = -np.inf if not I1 else max((fj - f_vals[i]) / (dj - d[i]) for i in I1)
        right_slope = np.inf if not I2 else min((f_vals[i] - fj) / (d[i] - dj) for i in I2)

        if left_slope > right_slope:
            continue

        # Check condition (8) or (9)
        f_min_abs = abs(f_min) if f_min != 0 else 1
        if I2 and f_min != 0:
            condition = eps <= (f_min - fj) / f_min_abs + (dj / f_min_abs) * right_slope
        elif f_min == 0:
            condition = fj <= dj * right_slope if I2 else True
        else:
            condition = True

        if condition:
            potential.append(rj)

    return potential

def DIRECT(f, bounds, max_iter=3000, max_evals=3000, eps=1e-4):
    n = bounds.shape[0]
    lower, length = normalize_bounds(bounds)

    def f_normalized(x):
        x_orig = denormalize_point(x, lower, length)
        return f(x_orig)

    # Initialization
    center = np.full(n, 0.5)
    f_min = f_normalized(center)
    best_point = center.copy()

    rectangles = []
    index_counter = 0

    # Initial hypercube edges = 1 in each dimension (normalized)
    edge_lengths = np.ones(n)

    rectangles.append(Rectangle(center, edge_lengths, f_min, index_counter))
    index_counter += 1

    delta = 1.0 / 3

    # Initial function evaluations at c1 ± delta*e_i
    for i in range(n):
        for direction in [+1, -1]:
            point = center.copy()
            point[i] += direction * delta
            if 0 <= point[i] <= 1:
                val = f_normalized(point)
                if val < f_min:
                    f_min = val
                    best_point = point.copy()
                rectangles.append(Rectangle(point, edge_lengths / 3, val, index_counter))
                index_counter += 1

    eval_count = 1
    iteration = 0

    KNOWN_GLOBAL_MIN = -186.7309088

    while iteration < max_iter and eval_count < max_evals:
        iteration += 1

        potential_rects = find_potentially_optimal_rectangles(rectangles, f_min, eps)

        if not potential_rects:
            break

        new_rects = []

        for rect in potential_rects:
            longest_edge = np.max(rect.edge_lengths)
            max_dims = np.where(rect.edge_lengths == longest_edge)[0]

            delta = longest_edge / 3

            # Calculate wj values for each maximal dimension
            wj = []
            f_plus = []
            f_minus = []

            for j in max_dims:
                p_plus = rect.center.copy()
                p_minus = rect.center.copy()

                p_plus[j] += delta
                p_minus[j] -= delta

                val_plus = f_normalized(p_plus) if 0 <= p_plus[j] <= 1 else np.inf
                val_minus = f_normalized(p_minus) if 0 <= p_minus[j] <= 1 else np.inf

                # Count function evals
                if val_plus != np.inf:
                    eval_count += 1
                    if val_plus < f_min:
                        f_min = val_plus
                        best_point = p_plus.copy()

                if val_minus != np.inf:
                    eval_count += 1
                    if val_minus < f_min:
                        f_min = val_minus
                        best_point = p_minus.copy()

                wj.append(min(val_plus, val_minus))

            # Sort dimensions by wj ascending
            sorted_dims = max_dims[np.argsort(wj)]

            # Divide rect into smaller rectangles along sorted dimensions
            # For each dimension, split into 3 sub-rectangles
            parents = [rect]

            for dim in sorted_dims:
                children = []
                for parent in parents:
                    d = np.max(parent.edge_lengths) / 3

                    centers = [parent.center.copy(),
                               parent.center.copy(),
                               parent.center.copy()]
                    centers[1][dim] += d
                    centers[2][dim] -= d

                    new_edge_lengths = parent.edge_lengths.copy()
                    new_edge_lengths[dim] /= 3

                    for c in centers:
                        if np.all(c >= 0) and np.all(c <= 1):
                            val = f_normalized(c)
                            eval_count += 1
                            if val < f_min:
                                f_min = val
                                best_point = c.copy()
                            children.append(Rectangle(c, new_edge_lengths, val, index_counter))
                            index_counter += 1
                parents = children
            new_rects.extend(parents)

        # Remove divided rectangles
        rectangles = [r for r in rectangles if r not in potential_rects]

        # Add new rectangles
        rectangles.extend(new_rects)

        if f_min <= KNOWN_GLOBAL_MIN:
            print(f"Known global minimum reached with value {f_min:.6f} at iteration {iteration}")
            break

    return best_point, f_min, eval_count, iteration

if __name__ == "__main__":
    bounds = np.array([[0, 1], [0, 1]])
    best_point, f_min, evals, iterations = DIRECT(scaled_shubert, bounds)
    print(f"\nBest point found (normalized): {best_point}")
    print(f"Function value at best point: {f_min}")
    print(f"Total function evaluations: {evals}")
    print(f"Total iterations: {iterations}")

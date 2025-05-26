import numpy as np
import matplotlib.pyplot as plt

class Rectangle:
    def __init__(self, center, edge_lengths, f_center, index):
        self.center = center
        self.edge_lengths = edge_lengths
        self.f_center = f_center
        self.index = index
        self.measure_val = 0.5 * np.linalg.norm(self.edge_lengths)

    def measure(self):
        return self.measure_val

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
    x = 20 * x_scaled - 10
    return shubert(x)

def find_potentially_optimal_rectangles(rectangles, f_min, eps):
    if not rectangles:
        return []
    potential = []
    d = np.array([r.measure() for r in rectangles])
    f_vals = np.array([r.f_center for r in rectangles])
    for j, rj in enumerate(rectangles):
        dj, fj = d[j], f_vals[j]
        left_slope = -np.inf
        right_slope = np.inf
        condition1 = True
        for i, ri in enumerate(rectangles):
            if i == j:
                continue
            di, fi = d[i], f_vals[i]
            if di < dj:
                slope = (fj - fi) / (dj - di)
                left_slope = max(left_slope, slope)
            elif di > dj:
                slope = (fi - fj) / (di - dj)
                right_slope = min(right_slope, slope)
            elif fi < fj:
                condition1 = False
                break
        if not condition1 or left_slope > right_slope:
            continue
        f_min_abs = abs(f_min) if f_min != 0 else 1
        if f_min != 0:
            condition2 = eps <= (f_min - fj) / f_min_abs + (dj / f_min_abs) * right_slope
        else:
            condition2 = fj <= dj * right_slope
        if condition2:
            potential.append(rj)
    return potential

def DIRECT(f, bounds, max_iter=3000, max_evals=3000, eps=1e-4, visualize=True):
    n = bounds.shape[0]
    lower, length = normalize_bounds(bounds)
    evaluated_points = {}

    def f_normalized(x):
        key = tuple(np.round(x, 10))
        if key in evaluated_points:
            return evaluated_points[key]
        x_orig = denormalize_point(np.array(key), lower, length)
        val = f(x_orig)
        evaluated_points[key] = val
        return val

    center = np.full(n, 0.5)
    f_min = f_normalized(center)
    best_point = center.copy()

    rectangles = []
    index_counter = 0
    edge_lengths = np.ones(n)
    rectangles.append(Rectangle(center, edge_lengths, f_min, index_counter))
    index_counter += 1

    delta = 1.0 / 3
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

    eval_count = len(evaluated_points)
    iteration = 0
    KNOWN_GLOBAL_MIN = np.float64(-186.7309088)
    all_evaluated = [np.array(k) for k in evaluated_points.keys()]

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
            wj = []
            for j in max_dims:
                p_plus = rect.center.copy()
                p_minus = rect.center.copy()
                p_plus[j] += delta
                p_minus[j] -= delta
                val_plus = f_normalized(p_plus) if 0 <= p_plus[j] <= 1 else np.inf
                val_minus = f_normalized(p_minus) if 0 <= p_minus[j] <= 1 else np.inf
                if val_plus < f_min:
                    f_min = val_plus
                    best_point = p_plus.copy()
                if val_minus < f_min:
                    f_min = val_minus
                    best_point = p_minus.copy()
                wj.append(min(val_plus, val_minus))
            sorted_dims = max_dims[np.argsort(wj)]
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
                            if val < f_min:
                                f_min = val
                                best_point = c.copy()
                            children.append(Rectangle(c, new_edge_lengths, val, index_counter))
                            index_counter += 1
                parents = children
            new_rects.extend(parents)
        rectangles = [r for r in rectangles if r not in potential_rects]
        rectangles.extend(new_rects)
        eval_count = len(evaluated_points)
        all_evaluated = [np.array(k) for k in evaluated_points.keys()]
        if f_min <= KNOWN_GLOBAL_MIN:
            print(f"Pasiektas globalus minimumas: {f_min:.6f} iteracijoje {iteration}")
            break

    if visualize and n == 2:
        plot_visualization(f, all_evaluated, best_point)

    return best_point, f_min, eval_count, iteration

def plot_visualization(f, points, best_point):
    X, Y = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    Z = np.array([[f(np.array([x, y])) for x in np.linspace(0, 1, 200)] for y in np.linspace(0, 1, 200)])
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=50, cmap='plasma')
    pts = np.array(points)
    plt.scatter(pts[:, 0], pts[:, 1], s=10, color='white', label='Įvertinti taškai')
    plt.plot(best_point[0], best_point[1], 'ro', label='Geriausias taškas', markersize=10)
    plt.colorbar(label='f(x)')
    plt.title("DIRECT optimizavimo vizualizacija")
    plt.xlabel('x1 (normalizuotas)')
    plt.ylabel('x2 (normalizuotas)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    bounds = np.array([[0, 1], [0, 1]])
    best_point, f_min, evals, iterations = DIRECT(scaled_shubert, bounds)
    print(f"\nGeriausias taškas (normalizuotas): {best_point}")
    print(f"Funkcijos reikšmė geriausiam taške: {f_min}")
    print(f"Funkcijos kvietimų: {evals}")
    print(f"Iteracijų skaičius: {iterations}")

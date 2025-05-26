import numpy as np

class Rectangle:
    def __init__(self, center, edge_lengths, f_center, index):
        self.center = center                # Stačiakampio centro taškas (normalizuotoj erdvėj)
        self.edge_lengths = edge_lengths    # Kraštinių ilgis
        self.f_center = f_center            # Funkcijos reikšmė centre
        self.index = index                  # Stačiakampio indentifikacija 

    def measure(self):
        # Atstumas nuo centro iki kraštinių yra pusė įsrižainės ilgio
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

        # Patikrinti 1 sąlygą: fj <= f_i visiems i c I3
        if not all(fj <= f_vals[i] for i in I3):
            continue

        # Suskaičiuoti k neligybėms
        left_slope = -np.inf if not I1 else max((fj - f_vals[i]) / (dj - d[i]) for i in I1)
        right_slope = np.inf if not I2 else min((f_vals[i] - fj) / (d[i] - dj) for i in I2)

        if left_slope > right_slope:
            continue

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

    # Inicializacija
    center = np.full(n, 0.5)
    f_min = f_normalized(center)
    best_point = center.copy()

    rectangles = []
    index_counter = 0

    # pradinis hiberkubo kraštinių ilgis = 1 (normalizuotas)
    edge_lengths = np.ones(n)

    rectangles.append(Rectangle(center, edge_lengths, f_min, index_counter))
    index_counter += 1

    delta = 1.0 / 3

    # Pradinės funkcijų reikšmių skaičiavimas: c1 ± delta*e_i
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

    KNOWN_GLOBAL_MIN = np.float64(-186.7309088)

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

            # Suskaičiuoti wj reikšmes kiekvienai maksimaliai ašiai
            for j in max_dims:
                p_plus = rect.center.copy()
                p_minus = rect.center.copy()

                p_plus[j] += delta
                p_minus[j] -= delta

                val_plus = f_normalized(p_plus) if 0 <= p_plus[j] <= 1 else np.inf
                val_minus = f_normalized(p_minus) if 0 <= p_minus[j] <= 1 else np.inf

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

            # Surikiuoti ašis pagal wj reikšmes
            sorted_dims = max_dims[np.argsort(wj)]

            parents = [rect]

            # Kiekvieną ašį padalinti į tris stačiakampius, pagal rikiavimą
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

        # Ištrinti padalintus stačiakampius
        rectangles = [r for r in rectangles if r not in potential_rects]

        # Pridėti naujus stačiakampius
        rectangles.extend(new_rects)

        if f_min <= KNOWN_GLOBAL_MIN:
            print(f"Pasiektas globalaus minimumo taškas: {f_min:.6f} iteracijoje: {iteration}")
            break

    return best_point, f_min, eval_count, iteration

if __name__ == "__main__":
    bounds = np.array([[0, 1], [0, 1]])
    best_point, f_min, evals, iterations = DIRECT(scaled_shubert, bounds)
    print(f"\nGeriausias taškas (normalizuotas): {best_point}")
    print(f"Funkcijos reikšmė geriausiam taške: {f_min}")
    print(f"Funckijos kvietimų: {evals}")
    print(f"Iteracijų skaičius: {iterations}")

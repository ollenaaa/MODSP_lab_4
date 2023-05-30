import math
import random
import numpy
import warnings
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


def generate_sequence(N, n, A, phi):
    x = np.linspace(0, 3, N)
    noise = np.random.uniform(-0.05 * A, 0.05 * A, N)
    y = A * np.sin(n * x + phi) + noise
    return y, x


def arithmetic_mean(sequence):
    return sum(sequence) / len(sequence)


def harmonic_mean(sequence):
    return len(sequence) / sum(1 / x for x in sequence)


def geometric_mean(sequence):
    a = [x for x in sequence if x != 0]
    array = np.array(a)
    return array.prod() ** (1 / len(array))


def _plot(x, y, y2):
    plt.plot(x, y)
    plt.plot(x, y2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def exact_value(A, n, phi, x):
    return [A * math.sin(n * xi + phi) for xi in x]


def compare_values(approximate, exact):
    absolute_error = math.fabs(approximate - exact)
    if numpy.abs(exact) == 0:
        relative_error = 0
    else:
        relative_error = absolute_error / numpy.abs(exact)
    max_absolute_error = numpy.max(absolute_error)
    max_relative_error = numpy.max(relative_error)
    min_absolute_error = numpy.min(absolute_error)
    min_relative_error = numpy.min(relative_error)
    return relative_error, absolute_error, max_relative_error, max_absolute_error, min_relative_error, min_absolute_error


N = 2100
n = 21
A = 1
phi = math.pi / 4

sequences, x = generate_sequence(N, n, A, phi)

a_mean_val = arithmetic_mean(sequences)
ex_a_val = arithmetic_mean(exact_value(A, n, phi, x))
rel_a_error, abs_a_error, max_rel_a_error, max_abs_a_error, min_rel_a_error, min_abs_a_error = compare_values(a_mean_val, ex_a_val)
print("Approximate arithmetic mean: " + str(a_mean_val) +
      "\nExact arithmetic mean: " + str(ex_a_val) +
      "\nMax absolute arithmetic error: " + str(max_abs_a_error) +
      "\nMax relative arithmetic error: " + str(max_rel_a_error) +
      "\nMin absolute arithmetic error: " + str(min_abs_a_error) +
      "\nMin relative arithmetic error: " + str(min_rel_a_error))

h_mean_val = harmonic_mean(sequences)
ex_h_val = harmonic_mean(exact_value(A, n, phi, x))
rel_h_error, abs_h_error, max_rel_h_error, max_abs_h_error, min_rel_h_error, min_abs_h_error = compare_values(h_mean_val, ex_h_val)
print("\n\nApproximate harmonic mean: " + str(h_mean_val) +
      "\nExact harmonic mean: " + str(ex_h_val) +
      "\nMax absolute harmonic error: " + str(max_abs_h_error) +
      "\nMax relative harmonic error: " + str(max_rel_h_error) +
      "\nMin absolute harmonic error: " + str(min_abs_h_error) +
      "\nMin relative harmonic error: " + str(min_rel_h_error))

g_mean_val = geometric_mean(sequences)
ex_g_val = geometric_mean(exact_value(A, n, phi, x))
rel_g_error, abs_g_error, max_rel_g_error, max_abs_g_error, min_rel_g_error, min_abs_g_error = compare_values(g_mean_val, ex_g_val)
print("\nApproximate geometric mean: " + str(g_mean_val) +
      "\nExact geometric mean: " + str(ex_g_val) +
      "\nMax absolute geometric error: " + str(max_abs_g_error) +
      "\nMax relative geometric error: " + str(max_rel_g_error) +
      "\nMin absolute geometric error: " + str(min_abs_g_error) +
      "\nMin relative geometric error: " + str(min_rel_g_error))

_plot(x, sequences, exact_value(A, n, phi, x))
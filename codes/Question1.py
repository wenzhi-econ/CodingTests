#! python3

"""
This file completes question 1 tasks of the coding test.

Wang Wenzhi 
Time: 2024-10-25
"""


# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??
# ?? step 0. import necessary packages
# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??

# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
# -? s-0-1. working directory paths
# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
import sys
import os

wd_path = r"E:\\RA\\BoothTests"
results_path = os.path.join(wd_path, "results")
sys.path.append(wd_path)

# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
# -? s-0-2. Other necessary packages
# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numpy.typing import NDArray

np.set_printoptions(threshold=1000, linewidth=150)
pd.set_option("mode.copy_on_write", True)


class ExpDistSimulation:
    """
    This class conducts exercises required in question 1.1 - question 1.4.
    """

    def __init__(
        self,
        lambda_value: float = 1 / 8,
        n_sample: int = 500,
        n_simulation: int = 1000,
    ):
        """
        Initialize an object instance with distributional and simulation pars.
        """
        self.n_sample = n_sample
        self.n_simulation = n_simulation
        self.lambda_value = lambda_value

    def __repr__(self) -> str:
        """
        Print the distributional and simulation parameters.
        """
        return (
            f"An exponential distribution simulation exercise with lambda = {self.lambda_value:.3f}, "
            f"sample size = {self.n_sample:.0f}, "
            f"and number of simulations = {self.n_simulation:.0f}.\n"
        )

    def gen_exp_sample(self, seed: int = 1234) -> NDArray:
        """
        This method generates a random draw from an exponential distribution.

        Parameters:
        self.lambda_value: the inverse of the expectation of the distribution
        self.n_sample: sample size
        seed: randomization seed

        Returns:
        sample: np.ndarray with shape (self.n_sample,)
        """
        scale_value = 1 / self.lambda_value
        rng = np.random.default_rng(seed=seed)
        sample = rng.exponential(scale=scale_value, size=self.n_sample)
        return sample

    def cal_sample_mean(self, array: NDArray) -> float:
        """
        This method calculates the sample mean from any sample array.

        Parameters:
        array: a np.ndarray with shape (self.n_sample,)
        """
        return array.mean()

    def adj_sample_mean(self, array: NDArray) -> float:
        """
        This method calculates the adjusted sample mean from any sample array
        drawn from an exponential distribution.

        Parameters:
        self.lambda_value: the inverse of the expectation of the distribution
        self.n_sample: sample size
        array: a np.ndarray with shape (self.n_sample,)
        """
        raw_sample_mean = self.cal_sample_mean(array)
        expectation = 1 / self.lambda_value
        res = np.sqrt(self.n_sample) * (raw_sample_mean - expectation)
        return res

    def simulation(self) -> NDArray:
        """
        This method conducts simulations. In particular, self.n_simulation
        random samples will be generated using the self.gen_exp_sample()
        method. Then all adjusted sample means calculated by the
        self.adj_sample_mean() method will stored in a np.ndarray.

        Parameters:
        self.lambda_value: the inverse of the expectation of the distribution
        self.n_sample: sample size
        self.n_simulation: number of simulations

        Returns:
        res: a np.ndarray with shape (self.n_simulation,)
        """
        res = np.zeros(shape=(self.n_simulation,))
        for i in range(self.n_simulation):
            sample = self.gen_exp_sample(seed=i)
            res_i = self.adj_sample_mean(array=sample)
            res[i] = res_i
        return res

    def simulation_plot(self, file_name, **kwargs):
        """
        This method plots the empirical distribution of the simulated adjusted
        sample mean (calculated using different self.n_sample), and saves the
        figure into a png file.
        """
        simulated_means = self.simulation()
        plt.hist(simulated_means, density=True, **kwargs)
        plt.title("Distribution of Simulated Adjusted Means")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.text(
            0.95,
            0.95,
            f"Sample Size: {self.n_sample}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=plt.gca().transAxes,
        )
        final_file_path = os.path.join(results_path, f"{file_name}.png")
        plt.savefig(final_file_path)
        plt.show()
        plt.close()


# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??
# ?? question 1.1.
# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??

ExpDistSimulation_5 = ExpDistSimulation(n_sample=5, lambda_value=0.125)
ExpDistSimulation_20 = ExpDistSimulation(n_sample=20, lambda_value=0.125)
ExpDistSimulation_50 = ExpDistSimulation(n_sample=50, lambda_value=0.125)
ExpDistSimulation_100 = ExpDistSimulation(n_sample=100, lambda_value=0.125)
ExpDistSimulation_500 = ExpDistSimulation(n_sample=500, lambda_value=0.125)

sample_5 = ExpDistSimulation_5.gen_exp_sample()
sample_5

sample_20 = ExpDistSimulation_20.gen_exp_sample()
sample_50 = ExpDistSimulation_50.gen_exp_sample()
sample_100 = ExpDistSimulation_100.gen_exp_sample()
sample_500 = ExpDistSimulation_500.gen_exp_sample()


# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??
# ?? question 1.2.
# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??

ExpDistSimulation_5.cal_sample_mean(sample_5)
ExpDistSimulation_20.cal_sample_mean(sample_20)
ExpDistSimulation_50.cal_sample_mean(sample_50)
ExpDistSimulation_100.cal_sample_mean(sample_100)
ExpDistSimulation_500.cal_sample_mean(sample_500)

# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??
# ?? question 1.3.
# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??

ExpDistSimulation_5.adj_sample_mean(sample_5)
ExpDistSimulation_20.adj_sample_mean(sample_20)
ExpDistSimulation_50.adj_sample_mean(sample_50)
ExpDistSimulation_100.adj_sample_mean(sample_100)
ExpDistSimulation_500.adj_sample_mean(sample_500)

# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??
# ?? question 1.4.
# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??

ExpDistSimulation_5.simulation_plot(
    bins=30,
    edgecolor="black",
    file_name="SimulatedAdjMeans_SampleSize5",
)

ExpDistSimulation_20.simulation_plot(
    bins=30,
    edgecolor="black",
    file_name="SimulatedAdjMeans_SampleSize20",
)

ExpDistSimulation_50.simulation_plot(
    bins=30,
    edgecolor="black",
    file_name="SimulatedAdjMeans_SampleSize50",
)

ExpDistSimulation_100.simulation_plot(
    bins=30,
    edgecolor="black",
    file_name="SimulatedAdjMeans_SampleSize100",
)

ExpDistSimulation_500.simulation_plot(
    bins=30,
    edgecolor="black",
    file_name="SimulatedAdjMeans_SampleSize500",
)


# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??
# ?? question 1.5.
# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??

sample_for_est = ExpDistSimulation(n_sample=5000).gen_exp_sample()


def obj_func(lambda_value: float, sample: NDArray) -> float:
    moment = 5000 * np.log(lambda_value) - lambda_value * np.sum(sample)
    obj = -moment
    return obj


bounds = [(1e-6, 100)]
initial_guess = 0.2
result = minimize(
    obj_func, x0=initial_guess, args=(sample_for_est,), bounds=bounds
)
print(
    f"The lambda value that maximizes the objective function is: {result.x[0]}."
)


# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
# -? question 1.5. constraints [0.5, 1]
# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?

initial_guess = 0.6
bounds_1 = [(0.5, 1)]

result_b1 = minimize(
    obj_func, x0=initial_guess, args=(sample_for_est,), bounds=bounds_1
)
print(
    f"The lambda value that maximizes the objective function is: {result_b1.x}."
)

# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
# -? question 1.5. constraints [0.2, 1]
# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?

bounds_2 = [(0.2, 1)]

result_b2 = minimize(
    obj_func, x0=initial_guess, args=(sample_for_est,), bounds=bounds_2
)
print(
    f"The lambda value that maximizes the objective function is: {result_b2.x}."
)

# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?
# -? question 1.5. constraints [0.1, 1]
# -?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?#-?

bounds_3 = [(0.1, 1)]

result_b3 = minimize(
    obj_func, x0=initial_guess, args=(sample_for_est,), bounds=bounds_3
)
print(
    f"The lambda value that maximizes the objective function is: {result_b3.x}."
)

# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??
# ?? question 1.7.
# ??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??#??

bounds = [(1e-6, 100)]

initial_guess_array = np.linspace(1e-6, 100, 10000)
results = np.zeros(shape=(10000))

i = 0
for guess in initial_guess_array:
    result_guess = minimize(
        obj_func, x0=guess, args=(sample_for_est,), bounds=bounds
    )
    lambda_estimated = result_guess.x[0]
    # print(result_guess.x[0])
    results[i] = lambda_estimated
    i = i + 1

plt.plot(initial_guess_array, results)
mean_result = np.mean(results)
plt.ylim(mean_result - 0.01, mean_result + 0.01)
plt.xlabel("Initial value")
plt.ylabel("Results")

final_file_path = os.path.join(results_path, "initial_guess_plot.png")
plt.savefig(final_file_path)

plt.plot(initial_guess_array, results)
plt.xlabel("Initial value")
plt.ylabel("Results")
final_file_path = os.path.join(
    results_path, "initial_guess_plot_rescaling.png"
)
plt.savefig(final_file_path)

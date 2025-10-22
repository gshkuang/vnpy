from collections.abc import Callable
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from random import random, choice
from time import perf_counter
from multiprocessing import get_context
from multiprocessing.context import BaseContext
from multiprocessing.managers import DictProxy
from _collections_abc import dict_keys, dict_values, Iterable

from tqdm import tqdm
from deap import creator, base, tools, algorithms  # type: ignore

import optuna

from .locale import _

OUTPUT_FUNC = Callable[[str], None]
EVALUATE_FUNC = Callable[[dict], dict]
KEY_FUNC = Callable[[tuple], float]


# Create individual class used in genetic algorithm optimization
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


class OptimizationSetting:
    """
    Setting for runnning optimization.
    """

    def __init__(self) -> None:
        """"""
        self.params: dict[str, list] = {}
        self.target_name: str = ""

    def add_parameter(
        self,
        name: str,
        start: float,
        end: float | None = None,
        step: float | None = None,
    ) -> tuple[bool, str]:
        """"""
        if end is None or step is None:
            self.params[name] = [start]
            return True, _("固定参数添加成功")

        if start >= end:
            return False, _("参数优化起始点必须小于终止点")

        if step <= 0:
            return False, _("参数优化步进必须大于0")

        value: float = start
        value_list: list[float] = []

        while value <= end:
            value_list.append(value)
            value += step

        self.params[name] = value_list

        return True, _("范围参数添加成功，数量{}").format(len(value_list))

    def set_target(self, target_name: str) -> None:
        """"""
        self.target_name = target_name

    def generate_settings(self) -> list[dict]:
        """"""
        keys: dict_keys = self.params.keys()
        values: dict_values = self.params.values()
        products: list = list(product(*values))

        settings: list = []
        for p in products:
            setting: dict = dict(zip(keys, p, strict=False))
            settings.append(setting)

        return settings


def check_optimization_setting(
    optimization_setting: OptimizationSetting, output: OUTPUT_FUNC = print
) -> bool:
    """"""
    if not optimization_setting.generate_settings():
        output(_("优化参数组合为空，请检查"))
        return False

    if not optimization_setting.target_name:
        output(_("优化目标未设置，请检查"))
        return False

    return True


def run_bf_optimization(
    evaluate_func: EVALUATE_FUNC,
    optimization_setting: OptimizationSetting,
    key_func: KEY_FUNC,
    max_workers: int | None = None,
    output: OUTPUT_FUNC = print,
) -> list[tuple]:
    """Run brutal force optimization"""
    settings: list[dict] = optimization_setting.generate_settings()

    output(_("开始执行穷举算法优化"))
    output(_("参数优化空间：{}").format(len(settings)))

    start: float = perf_counter()

    with ProcessPoolExecutor(max_workers, mp_context=get_context("spawn")) as executor:
        it: Iterable = tqdm(executor.map(evaluate_func, settings), total=len(settings))
        results: list[tuple] = list(it)
        results.sort(reverse=True, key=key_func)

        end: float = perf_counter()
        cost: int = int(end - start)
        output(_("穷举算法优化完成，耗时{}秒").format(cost))

        return results


def run_ga_optimization(
    evaluate_func: EVALUATE_FUNC,
    optimization_setting: OptimizationSetting,
    key_func: KEY_FUNC,
    max_workers: int | None = None,
    pop_size: int = 100,  # population size: number of individuals in each generation
    ngen: int = 30,  # number of generations: number of generations to evolve
    mu: (
        int | None
    ) = None,  # mu: number of individuals to select for the next generation
    lambda_: (
        int | None
    ) = None,  # lambda: number of children to produce at each generation
    cxpb: float = 0.95,  # crossover probability: probability that an offspring is produced by crossover
    mutpb: (
        float | None
    ) = None,  # mutation probability: probability that an offspring is produced by mutation
    indpb: float = 1.0,  # independent probability: probability for each gene to be mutated
    output: OUTPUT_FUNC = print,
) -> list[tuple]:
    """Run genetic algorithm optimization"""
    # Define functions for generate parameter randomly
    settings: list[dict] = optimization_setting.generate_settings()
    parameter_tuples: list[list[tuple]] = [list(d.items()) for d in settings]

    def generate_parameter() -> list:
        """"""
        return choice(parameter_tuples)

    def mutate_individual(individual: list, indpb: float) -> tuple:
        """"""
        size: int = len(individual)
        paramlist: list = generate_parameter()
        for i in range(size):
            if random() < indpb:
                individual[i] = paramlist[i]
        return (individual,)

    # Set up multiprocessing Pool and Manager
    ctx: BaseContext = get_context("spawn")
    with ctx.Manager() as manager, ctx.Pool(max_workers) as pool:
        # Create shared dict for result cache
        cache: DictProxy[tuple, tuple] = manager.dict()

        # Set up toolbox
        toolbox: base.Toolbox = base.Toolbox()
        toolbox.register(
            "individual", tools.initIterate, creator.Individual, generate_parameter
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", mutate_individual, indpb=indpb)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("map", pool.map)
        toolbox.register("evaluate", ga_evaluate, cache, evaluate_func, key_func)

        # Set default values for DEAP parameters if not specified
        if mu is None:
            mu = int(pop_size * 0.8)

        if lambda_ is None:
            lambda_ = pop_size

        if mutpb is None:
            mutpb = 1.0 - cxpb

        total_size: int = len(parameter_tuples)
        pop: list = toolbox.population(pop_size)

        # Run ga optimization
        output(_("开始执行遗传算法优化"))
        output(_("参数优化空间：{}").format(total_size))
        output(_("每代族群总数：{}").format(pop_size))
        output(_("优良筛选个数：{}").format(mu))
        output(_("迭代次数：{}").format(ngen))
        output(_("交叉概率：{:.0%}").format(cxpb))
        output(_("突变概率：{:.0%}").format(mutpb))
        output(_("个体突变概率：{:.0%}").format(indpb))

        start: float = perf_counter()

        algorithms.eaMuPlusLambda(
            pop, toolbox, mu, lambda_, cxpb, mutpb, ngen, verbose=True
        )

        end: float = perf_counter()
        cost: int = int(end - start)

        output(_("遗传算法优化完成，耗时{}秒").format(cost))

        results: list = list(cache.values())
        results.sort(reverse=True, key=key_func)
        return results


def run_optuna_optimization(
    evaluate_func: EVALUATE_FUNC,
    optimization_setting: OptimizationSetting,
    key_func: KEY_FUNC,
    n_trials: int = 100,
    timeout: int | None = None,
    direction: str = "maximize",
    output: OUTPUT_FUNC = print,
) -> list[tuple]:
    # Check if optimization setting is valid
    settings: list[dict] = optimization_setting.generate_settings()
    if not settings:
        output(_("优化参数组合为空，请检查"))
        return []

    # Convert OptimizationSetting to Optuna parameter space
    param_space: dict = {}
    for param_name, param_values in optimization_setting.params.items():
        if len(param_values) == 1:
            # Fixed parameter
            param_space[param_name] = param_values
        else:
            # Range parameter - determine if it's continuous or discrete
            if all(isinstance(x, (int, float)) for x in param_values):
                # Numeric parameter
                param_space[param_name] = [min(param_values), max(param_values)]
            else:
                # Categorical parameter
                param_space[param_name] = param_values

    output(_("开始执行Optuna算法优化"))
    output(_("参数优化空间：{}").format(param_space))
    output(_("最大试验次数：{}").format(n_trials))

    # Store best result for comparison
    best_result_storage = {
        "result": None,
        "value": float("-inf") if direction == "maximize" else float("inf"),
    }

    def objective(trial):
        """Optuna objective function"""
        current_params = {}

        # Generate parameters for each trial
        for param_name, param_range in param_space.items():
            if isinstance(param_range, list) and len(param_range) >= 2:
                if all(isinstance(x, (int, float)) for x in param_range):
                    # Numeric range parameter
                    if all(isinstance(x, int) for x in param_range):
                        # Integer parameter
                        current_params[param_name] = trial.suggest_int(
                            param_name, min(param_range), max(param_range)
                        )
                    else:
                        # Float parameter
                        current_params[param_name] = trial.suggest_float(
                            param_name, min(param_range), max(param_range)
                        )
                else:
                    # Categorical parameter
                    current_params[param_name] = trial.suggest_categorical(
                        param_name, param_range
                    )
            else:
                # Fixed parameter
                current_params[param_name] = (
                    param_range[0] if isinstance(param_range, list) else param_range
                )

        try:
            # Evaluate the parameters
            result = evaluate_func(current_params)
            metric_value = key_func(result)

            # Store best result
            if (
                direction == "maximize" and metric_value > best_result_storage["value"]
            ) or (
                direction == "minimize" and metric_value < best_result_storage["value"]
            ):
                best_result_storage["result"] = result
                best_result_storage["value"] = metric_value

            return metric_value

        except Exception as e:
            output(_("试验失败: {}").format(str(e)))
            # Return worst value for failed trials
            return float("-inf") if direction == "maximize" else float("inf")

    # Create Optuna study
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    start: float = perf_counter()

    # Run optimization
    study.optimize(
        objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True
    )

    end: float = perf_counter()
    cost: int = int(end - start)

    output(_("Optuna算法优化完成，耗时{}秒").format(cost))
    output(_("最佳目标值: {:.4f}").format(study.best_value))
    output(_("总试验次数: {}").format(len(study.trials)))
    output(_("最佳参数: {}").format(study.best_params))

    # Show parameter importance if enough trials
    if len(study.trials) >= 10:
        try:
            importance = optuna.importance.get_param_importances(study)
            importance = {k: round(float(v), 4) for k, v in importance.items()}
            output(_("参数重要性: {}").format(importance))
        except Exception:
            pass

    # Convert results to the same format as other optimization methods
    results = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            # Create result tuple: (parameters_dict, target_value, full_result_dict)
            param_dict = trial.params
            target_value = trial.value

            # Try to get the full result dict, fallback to creating a minimal one
            if (
                best_result_storage["result"]
                and abs(target_value - best_result_storage["value"]) < 1e-10
            ):
                full_result = best_result_storage["result"]
            else:
                # Create a minimal result dict with the target value
                full_result = {optimization_setting.target_name: target_value}

            results.append((param_dict, target_value, full_result))

    # Sort results by target value (descending for maximize, ascending for minimize)
    results.sort(reverse=(direction == "maximize"), key=lambda x: x[1])

    return results


def ga_evaluate(
    cache: dict, evaluate_func: Callable, key_func: Callable, parameters: list
) -> tuple[float,]:
    """
    Functions to be run in genetic algorithm optimization.
    """
    tp: tuple = tuple(parameters)
    if tp in cache:
        result: dict = cache[tp]
    else:
        setting: dict = dict(parameters)
        result = evaluate_func(setting)
        cache[tp] = result

    value: float = key_func(result)
    return (value,)

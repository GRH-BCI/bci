import datetime
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, Future
from typing import Optional, Tuple, Type, Callable, List, Set

from optuna.study._optimize import _optimize_sequential
from optuna.trial import FrozenTrial


def optimize_parallel(
    study: "optuna.Study",
    func: "optuna.study.study.ObjectiveFuncType",
    n_trials: Optional[int] = None,
    timeout: Optional[float] = None,
    n_jobs: int = -1,
    catch: Tuple[Type[Exception], ...] = (),
    callbacks: Optional[List[Callable[["optuna.Study", FrozenTrial], None]]] = None,
    gc_after_trial: bool = False,
    show_progress_bar: bool = False,
) -> None:
    if not isinstance(catch, tuple):
        raise TypeError(
            "The catch argument is of type '{}' but must be a tuple.".format(type(catch).__name__)
        )

    if not study._optimize_lock.acquire(False):
        raise RuntimeError("Nested invocation of `Study.optimize` method isn't allowed.")

    study._stop_flag = False

    try:
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        time_start = datetime.datetime.now()
        futures: Set[Future] = set()

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for n_submitted_trials in itertools.count():
                if study._stop_flag:
                    break

                if (
                    timeout is not None
                    and (datetime.datetime.now() - time_start).total_seconds() > timeout
                ):
                    break

                if n_trials is not None and n_submitted_trials >= n_trials:
                    break

                if len(futures) >= n_jobs:
                    completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                    # Raise if exception occurred in executing the completed futures.
                    for f in completed:
                        f.result()

                futures.add(
                    executor.submit(
                        _optimize_sequential,
                        study,
                        func,
                        1,
                        timeout,
                        catch,
                        callbacks,
                        gc_after_trial,
                        True,
                        time_start,
                        None,
                    )
                )
    finally:
        study._optimize_lock.release()

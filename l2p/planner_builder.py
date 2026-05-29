"""
PDDL Planner module

This module defines the `Planner` abstract class and its implementations for executing
external automated planners on generated PDDL domain and problem instances.

Architecture:
- Stateless Execution: Planners return a standardized `PlanningResult` dataclass rather
  than storing state internally, allowing for safe parallel execution.
- Standardized Feedback: Both successful plans and crash traces (stderr) are captured
  and formatted to easily integrate into downstream LLM diagnostic and refinement loops.
- Flexible Backends: Supports command-line planners (e.g., Fast Downward) and native
  Python APIs (e.g., Unified Planning).

Optional Dependencies:
- `FastDownward` requires the Fast Downward executable path to be passed at runtime.
- `UnifiedPlanning` requires the `unified-planning` Python library and specific engine
  extensions (e.g., `pip install 'unified-planning[engines]'`).
"""

__all__ = ["PlanningResult", "Planner", "TimeoutException", "UnifiedPlanning", "FastDownward"]

import re
import subprocess
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class PlanningResult:
    is_successful: bool
    plan: Optional[List[str]] = None
    error_message: Optional[str] = None
    raw_output: str = ""
    metrics: Dict[str, Any] = field(
        default_factory=dict
    )  # e.g., search time, expanded nodes


# ---------------------------------------------------------------------------
# ABSTRACT PLANNER CLASS
# ---------------------------------------------------------------------------


class Planner(ABC):
    def __init__(
        self, executable_path: Optional[str] = None, cleanup_temp_files: bool = True
    ):
        """Base configuration for any external planner."""
        self.executable_path = executable_path
        self.cleanup_temp_files = cleanup_temp_files

    @abstractmethod
    def run_planner(
        self, domain_path: str, problem_path: Optional[str] = None, timeout: int = 60, **kwargs
    ) -> PlanningResult:
        """
        Executes the planner.
        Args:
            domain_path (str): Path to PDDL domain file
            problem_path (Optional[str]): Path to PDDL problem file
            timeout (int): Seconds before terminating planner process
            **kwargs: Planner specific arguments (e.g., search heuristic, memory limits)
        Returns:
            PlanningResult: A result object resulted from the planners output
        """
        raise NotImplementedError("This method should be overriden by subclasses.")

    @abstractmethod
    def parse_plan(self, raw_output: str) -> PlanningResult:
        """
        Translates specific planner's messy stdout into standardized PlanningResult class.
        Args:
            raw_output (str): The raw plan steps output
        Returns:
            PlanningResult: A clean result object containing clean plan
        """
        raise NotImplementedError("This method should be overriden by subclasses.")

    @abstractmethod
    def handle_error(
        self, stderr: str, returncode: Optional[int] = None
    ) -> PlanningResult:
        """
        Processes planner crashes, timeouts, memory limits, or PDDL syntax errors.
        Args:
            stderr (str): The standard error output or exception string from planner
            returncode (Optional[int]): The exit code of the planner subprocess
        Returns:
            PlanningResult: A failed result object containing parsed error message
        """
        raise NotImplementedError("This method should be overriden by subclasses.")


class TimeoutException(Exception):
    pass


def _run_with_timeout(func, timeout: int, *args, **kwargs):
    """Run *func* in a daemon thread with a timeout. Cross-platform (Unix + Windows)."""
    result = []
    exception = []

    def runner():
        try:
            result.append(func(*args, **kwargs))
        except Exception as e:
            exception.append(e)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        raise TimeoutException(f"Planner execution timed out after {timeout} seconds")
    if exception:
        raise exception[0]
    return result[0]


# ---------------------------------------------------------------------------
# UNIFIED PLANNING (UP) PLANNER
# ---------------------------------------------------------------------------


class UnifiedPlanning(Planner):
    def __init__(self, executable_path=None, cleanup_temp_files=True):
        super().__init__(executable_path, cleanup_temp_files)

        try:
            from unified_planning.io import PDDLReader
            from unified_planning.shortcuts import OneshotPlanner
        except ImportError:
            raise ImportError(
                "The 'unified_planning' library is required for running UP planner but is not installed. "
                "Install it using: `pip install unified-planning`"
            )

        self.reader = PDDLReader()
        self.planner = OneshotPlanner

    def run_planner(
        self,
        domain_path: str,
        problem_path: Optional[str] = None,
        engine: Optional[str] = None,
        timeout=60,
        **kwargs,
    ):
        """
        Executes the Unified Planning engine.
        Args:
            domain_path (str): Path to PDDL domain file
            problem_path (str): Path to PDDL problem file
            engine (str): The name of the UP engine to use (default: 'aries')
            timeout (int): Seconds before terminating planner process

        **Side Note** After installing unified-planning library, you must
            install specific planner: `pip install 'unified-planning[engine]'`

        Returns:
            PlanningResult: A result object resulted from the planners output
        """

        if not engine:
            engine = "aries"  # default (must still install `aries` engine!)

        try:
            if problem_path is not None:
                problem = self.reader.parse_problem(domain_path, problem_path)
            else:
                problem = self.reader.parse_problem(domain_path)

            def _solve():
                return self.planner(
                    name=engine, problem_kind=problem.kind, **kwargs
                ).solve(problem)

            if timeout:
                result = _run_with_timeout(_solve, timeout)
            else:
                result = _solve()

            if result.plan is not None:
                raw_plan_str = str(result.plan)
                return self.parse_plan(raw_plan_str)
            else:
                status_msg = f"UP engine `{engine}` failed to find a plan. Status: {result.status.name}"
                return self.handle_error(stderr=status_msg, returncode=None)

        except TimeoutException:
            return self.handle_error(
                stderr=f"Planner timed out after {timeout} seconds", returncode=-1
            )

        except Exception as e:
            return self.handle_error(
                stderr=f"Unified Planning execution crashed: {str(e)}", returncode=-2
            )

    def parse_plan(self, raw_output: str) -> PlanningResult:
        """
        Translates UP's string plan object into standardized list of action strings.
        Args:
            raw_output (str): The raw plan steps output
        Returns:
            PlanningResult: A clean result object containing clean plan
        """
        actions = []
        for line in raw_output.split("\n"):
            line = line.strip()
            if (
                line
                and not line.startswith("SequentialPlan")
                and not line.startswith("TimeTriggeredPlan")
            ):
                actions.append(line)

        return PlanningResult(is_successful=True, plan=actions, raw_output=raw_output)

    def handle_error(
        self, stderr: str, returncode: Optional[int] = None
    ) -> PlanningResult:
        if returncode == -1:
            error_prefix = "[TIMEOUT] "
        elif returncode == -2:
            error_prefix = "[SYSTEM CRASH] "
        else:
            error_prefix = "[PLANNER ERROR] "

        error_msg = error_prefix + stderr

        return PlanningResult(
            is_successful=False, error_message=error_msg, raw_output=stderr
        )


# ---------------------------------------------------------------------------
# FASTDOWNWARD (FD) PLANNER
# ---------------------------------------------------------------------------

# define the exit codes
SUCCESS = 0
SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY = 1
SEARCH_PLAN_FOUND_AND_OUT_OF_TIME = 2
SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY_AND_TIME = 3

TRANSLATE_UNSOLVABLE = 10
SEARCH_UNSOLVABLE = 11
SEARCH_UNSOLVED_INCOMPLETE = 12

TRANSLATE_OUT_OF_MEMORY = 20
TRANSLATE_OUT_OF_TIME = 21
SEARCH_OUT_OF_MEMORY = 22
SEARCH_OUT_OF_TIME = 23
SEARCH_OUT_OF_MEMORY_AND_TIME = 24

TRANSLATE_CRITICAL_ERROR = 30
TRANSLATE_INPUT_ERROR = 31
SEARCH_CRITICAL_ERROR = 32
SEARCH_INPUT_ERROR = 33
SEARCH_UNSUPPORTED = 34
DRIVER_CRITICAL_ERROR = 35
DRIVER_INPUT_ERROR = 36
DRIVER_UNSUPPORTED = 37


class FastDownward(Planner):
    def __init__(
        self, executable_path: Optional[str] = None, cleanup_temp_files: bool = True
    ):
        super().__init__(executable_path, cleanup_temp_files)

    def run_planner(self, domain_path: str, problem_path: str, timeout=60, **kwargs):
        """
        Executes the FastDownward engine.
        Args:
            domain_path (str): Path to PDDL domain file
            problem_path (str): Path to PDDL problem file
            timeout (int): Seconds before terminating planner process

        Returns:
            PlanningResult: A result object resulted from the planners output
        """
        custom_args = kwargs.get("custom_args", [])
        alias = kwargs.get("alias", "lama-first")

        if custom_args:
            # if the user provides custom args, append them after the domain and problem files
            # example CLI: fast-downward.py domain.pddl problem.pddl --search "astar(lmcut())"
            cmd = [self.executable_path, domain_path, problem_path] + custom_args
        else:
            # Otherwise, use the alias format BEFORE the domain and problem files
            # example CLI: fast-downward.py --alias lama-first domain.pddl problem.pddl
            cmd = [self.executable_path, "--alias", alias, domain_path, problem_path]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )

            # FastDownward comines stdout and stderr in weird ways so we keep both
            raw_output = result.stdout + "\n" + result.stderr

            if result.returncode == SUCCESS:
                return self.parse_plan(raw_output=raw_output)
            else:
                return self.handle_error(
                    stderr=raw_output, returncode=result.returncode
                )

        except subprocess.TimeoutExpired as e:
            return PlanningResult(
                is_successful=False,
                error_message=f"Planner timed out after {timeout} seconds.",
                raw_output=str(e),
            )
        except Exception as e:
            return PlanningResult(
                is_successful=False,
                error_message=f"Subprocess execution failed: {str(e)}",
                raw_output="",
            )

    def parse_plan(self, raw_output: str) -> PlanningResult:
        """
        Translates FD's string plan object into standardized list of action strings.
        Args:
            raw_output (str): The raw plan steps output
        Returns:
            PlanningResult: A clean result object containing clean plan
        """
        plan_steps_str = re.findall(r"^\w+.*\(.*\)", raw_output, re.MULTILINE)

        if plan_steps_str:
            plan_list = (
                plan_steps_str[0].split("\n")
                if isinstance(plan_steps_str, list)
                else []
            )
            return PlanningResult(
                is_successful=True, plan=plan_list, raw_output=raw_output
            )
        else:
            return PlanningResult(
                is_successful=False,
                error_message="Planner exited with SUCCESS, but no plan steps could be extracted from stdout.",
                raw_output=raw_output,
            )

    def handle_error(
        self, stderr: str, returncode: Optional[int] = None
    ) -> PlanningResult:
        """Maps FD's specific exit codes to human-readable error messages."""

        # for 'plan_found = False'
        error_map = {
            TRANSLATE_UNSOLVABLE: "Translate phase determined the problem is unsolvable.",
            SEARCH_UNSOLVABLE: "Search phase determined the problem is unsolvable.",
            SEARCH_UNSOLVED_INCOMPLETE: "Search phase was incomplete and did not solve the problem.",
            TRANSLATE_OUT_OF_MEMORY: "Translate phase ran out of memory.",
            TRANSLATE_OUT_OF_TIME: "Translate phase ran out of time.",
            SEARCH_OUT_OF_MEMORY: "Search phase ran out of memory.",
            SEARCH_OUT_OF_TIME: "Search phase ran out of time.",
            SEARCH_OUT_OF_MEMORY_AND_TIME: "Search phase ran out of memory and time.",
            TRANSLATE_CRITICAL_ERROR: "Critical error in translate phase. (Check PDDL syntax)",
            TRANSLATE_INPUT_ERROR: "Input error in translate phase. (Check PDDL syntax)",
            SEARCH_CRITICAL_ERROR: "Critical error in search phase.",
            SEARCH_INPUT_ERROR: "Input error in search phase.",
            SEARCH_UNSUPPORTED: "Search phase encountered an unsupported PDDL feature.",
            DRIVER_CRITICAL_ERROR: "Critical error in the driver.",
            DRIVER_INPUT_ERROR: "Input error in the driver.",
            DRIVER_UNSUPPORTED: "Driver encountered an unsupported feature.",
        }

        # handle 'plan_found = True' exit codes (e.g., LAMA portfolio found a sub-optimal plan but crashed later)
        partial_success_codes = {
            SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY: "Plan found but the search ran out of memory before finishing.",
            SEARCH_PLAN_FOUND_AND_OUT_OF_TIME: "Plan found but the search ran out of time before finishing.",
            SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY_AND_TIME: "Plan found but the search ran out of memory and time.",
        }

        if returncode in partial_success_codes:
            result = self.parse_plan(stderr)
            result.error_message = partial_success_codes[returncode]
            return result

        error_msg = error_map.get(
            returncode, f"Unknown error occurred with exit code: {returncode}"
        )

        return PlanningResult(
            is_successful=False, error_message=error_msg, raw_output=stderr
        )
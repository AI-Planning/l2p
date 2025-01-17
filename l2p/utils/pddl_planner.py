"""
L2P comes pre-packaged with FastDownward: https://www.fast-downward.org 

For usage, users must clone or download the submodule /downward separately and direct the 
`planner_path` to the folder. This module is not necessary to use L2P, but for ease of use
to produce plans from generated domain and problem PDDL specifications via LLMs.
"""

import subprocess, re

# Define the exit codes
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


class FastDownward:

    def __init__(self, planner_path: str):
        self.planner_path = planner_path  # directory of FastDownward planner

    def run_fast_downward(
        self, domain_file: str, problem_file: str, search_alg: str = "lama-first"
    ):
        """
        Main function to run planner.

        Args:
            - domain_file (str): PDDL domain file path
            - problem_file (str): PDDL problem file path
            - search_alg (str): search algorithm/heuristic to use
                + refer to: https://www.fast-downward.org/PlannerUsage

        Returns:
            - success (bool): if a plan was found, otherwise False for incomplete.
            - plan_output (str): plan output information.

        """
        try:
            result = subprocess.run(
                [self.planner_path, "--alias", search_alg, domain_file, problem_file],
                capture_output=True,
                text=True,
            )

            exitcodes = [result.returncode]

            if result.returncode == SUCCESS:
                # Planning succeeded
                print("Planning succeeded!")
                print(
                    "All run components successfully terminated (translator: completed, search: found a plan, validate: validated a plan)"
                )

                # Extract the plan steps from the output
                plan_output = self.extract_plan_steps(result.stdout)
                if plan_output:
                    return True, plan_output
                else:
                    return False, "No plan found in the output."
            else:
                # Planning failed
                exitcode, plan_found = self.generate_portfolio_exitcode(exitcodes)
                return False, self.handle_error(exitcode, plan_found)
        except Exception as e:
            print("An error occurred while running the planner.")
            return False, str(e)

    def extract_plan_steps(self, output):
        plan_steps = re.findall(r"^\w+.*\(.*\)", output, re.MULTILINE)
        return "\n".join(plan_steps)

    def handle_error(self, exitcode, plan_found):
        if plan_found:
            if exitcode == SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY:
                return "Plan found but the search ran out of memory."
            elif exitcode == SEARCH_PLAN_FOUND_AND_OUT_OF_TIME:
                return "Plan found but the search ran out of time."
            elif exitcode == SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY_AND_TIME:
                return "Plan found but the search ran out of memory and time."
            else:
                return f"Unknown plan occurred with exit code: {exitcode}"
        else:
            if exitcode == TRANSLATE_UNSOLVABLE:
                return "Translate phase determined the problem is unsolvable."
            elif exitcode == SEARCH_UNSOLVABLE:
                return "Search phase determined the problem is unsolvable."
            elif exitcode == SEARCH_UNSOLVED_INCOMPLETE:
                return "Search phase was incomplete and did not solve the problem."
            elif exitcode == TRANSLATE_OUT_OF_MEMORY:
                return "Translate phase ran out of memory."
            elif exitcode == TRANSLATE_OUT_OF_TIME:
                return "Translate phase ran out of time."
            elif exitcode == SEARCH_OUT_OF_MEMORY:
                return "Search phase ran out of memory."
            elif exitcode == SEARCH_OUT_OF_TIME:
                return "Search phase ran out of time."
            elif exitcode == SEARCH_OUT_OF_MEMORY_AND_TIME:
                return "Search phase ran out of memory and time."
            elif exitcode == TRANSLATE_CRITICAL_ERROR:
                return "Critical error in translate phase."
            elif exitcode == TRANSLATE_INPUT_ERROR:
                return "Input error in translate phase."
            elif exitcode == SEARCH_CRITICAL_ERROR:
                return "Critical error in search phase."
            elif exitcode == SEARCH_INPUT_ERROR:
                return "Input error in search phase."
            elif exitcode == SEARCH_UNSUPPORTED:
                return "Search phase encountered an unsupported feature."
            elif exitcode == DRIVER_CRITICAL_ERROR:
                return "Critical error in the driver."
            elif exitcode == DRIVER_INPUT_ERROR:
                return "Input error in the driver."
            elif exitcode == DRIVER_UNSUPPORTED:
                return "Driver encountered an unsupported feature."
            else:
                return f"Unknown error occurred with exit code: {exitcode}"

    def is_unrecoverable(self, exitcode):
        # Exit codes in the range from 30 to 39 represent unrecoverable failures.
        return 30 <= exitcode < 40

    def generate_portfolio_exitcode(self, exitcodes):

        print("Exit codes: {}".format(exitcodes))
        exitcodes = set(exitcodes)
        unrecoverable_codes = [
            code for code in exitcodes if self.is_unrecoverable(code)
        ]

        # There are unrecoverable exit codes.
        if unrecoverable_codes:
            print("Error: Unexpected exit codes: {}".format(unrecoverable_codes))
            if len(unrecoverable_codes) == 1:
                return (unrecoverable_codes[0], False)
            else:
                return (SEARCH_CRITICAL_ERROR, False)

        # At least one plan was found.
        if SUCCESS in exitcodes:
            if SEARCH_OUT_OF_MEMORY in exitcodes and SEARCH_OUT_OF_TIME in exitcodes:
                return (SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY_AND_TIME, True)
            elif SEARCH_OUT_OF_MEMORY in exitcodes:
                return (SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY, True)
            elif SEARCH_OUT_OF_TIME in exitcodes:
                return (SEARCH_PLAN_FOUND_AND_OUT_OF_TIME, True)
            else:
                return (SUCCESS, True)

        # A config proved unsolvability or did not find a plan.
        for code in [SEARCH_UNSOLVABLE, SEARCH_UNSOLVED_INCOMPLETE]:
            if code in exitcodes:
                return (code, False)

        # No plan was found due to hitting resource limits.
        if SEARCH_OUT_OF_MEMORY in exitcodes and SEARCH_OUT_OF_TIME in exitcodes:
            return (SEARCH_OUT_OF_MEMORY_AND_TIME, False)
        elif SEARCH_OUT_OF_MEMORY in exitcodes:
            return (SEARCH_OUT_OF_MEMORY, False)
        elif SEARCH_OUT_OF_TIME in exitcodes:
            return (SEARCH_OUT_OF_TIME, False)

        assert False, "Error: Unhandled exit codes: {}".format(exitcodes)

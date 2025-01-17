"""
This module contains the Domain class which contains all domains found in LLM+P
"""

import glob


def postprocess(x):
    return x.strip()


class Domain:
    def __init__(self, name):
        self.name = name
        self.context = ("p_example.nl", "p_example.pddl", "p_example.sol")
        self.tasks = []
        self.grab_tasks()

    def grab_tasks(self):
        path = f"paper_reconstructions/llm+p/domains/{self.name}/problems"
        nls = []
        for fn in glob.glob(f"{path}/*.nl"):

            fn_ = fn.split("/")[-1]
            nls.append(fn_)

        self.tasks = sorted(nls)

    def get_task_name(self, i):
        name = self.tasks[i]
        name = name.replace("nl", "pddl")
        return name

    def get_task_file(self, i):
        nl = self.tasks[i]
        return f"paper_reconstructions/llm+p/domains/{self.name}/problems/{nl}"

    def get_task(self, i):
        nl_f = self.get_task_file(i)

        with open(nl_f, "r") as f:
            nl = f.read()

        return postprocess(nl)

    def get_context(self):
        nl_f = f"paper_reconstructions/llm+p/domains/{self.name}/{self.context[0]}"
        pddl_f = f"paper_reconstructions/llm+p/domains/{self.name}/{self.context[1]}"
        sol_f = f"paper_reconstructions/llm+p/domains/{self.name}/{self.context[2]}"
        with open(nl_f, "r") as f:
            nl = f.read()
        with open(pddl_f, "r") as f:
            pddl = f.read()
        with open(sol_f, "r") as f:
            sol = f.read()
        return postprocess(nl), postprocess(pddl), postprocess(sol)

    def get_domain_pddl(self):
        domain_pddl_f = self.get_domain_pddl_file()
        with open(domain_pddl_f, "r") as f:
            domain_pddl = f.read()
        return postprocess(domain_pddl)

    def get_domain_pddl_file(self):
        domain_pddl_f = f"paper_reconstructions/llm+p/domains/{self.name}/domain.pddl"
        return domain_pddl_f

    def get_domain_nl(self):
        domain_nl_f = self.get_domain_nl_file()
        try:
            with open(domain_nl_f, "r") as f:
                domain_nl = f.read()
        except:
            domain_nl = "Nothing"
        return postprocess(domain_nl)

    def get_domain_nl_file(self):
        domain_nl_f = f"paper_reconstructions/llm+p/domains/{self.name}/domain.nl"
        return domain_nl_f


class Barman(Domain):
    name = "barman"


class Floortile(Domain):
    name = "floortile"


class Termes(Domain):
    name = "termes"


class Tyreworld(Domain):
    name = "tyreworld"


class Grippers(Domain):
    name = "grippers"


class Storage(Domain):
    name = "storage"


class Blocksworld(Domain):
    name = "blocksworld"


class Manipulation(Domain):
    name = "manipulation"

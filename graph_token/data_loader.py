from typing import Type
from enum import Enum, StrEnum

import torch
import networkx as nx
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import from_networkx

import tasks


class Difficulty(StrEnum):
    EASY = "easy"
    HARD = "hard"


class Tasks(Enum):
    CONNECTED = tasks.Connected_Task
    DIAMETER = tasks.Diameter_Task
    DISTANCE = tasks.Distance_Task
    GED = tasks.GED_Task
    MCP = tasks.MCP_Task
    MCS = tasks.MCS_Task
    MIS = tasks.MIS_Task
    MVC = tasks.MVC_Task
    NEIGHBOR = tasks.Neighbor_Task
    TSP = tasks.TSP_Task

    @property
    def task_class(self) -> Type[tasks.NPTask]:
        """Get the task class for this enum value."""
        return self._value_


class GPDataset(Dataset):
    def __init__(
        self,
        task: Tasks = Tasks.CONNECTED,
        difficulty: Difficulty = Difficulty.EASY,
        root=None,
        transform=None,
        pre_transform=None,
        dataset_loc="dataset",
    ):
        self.difficulty = difficulty
        super().__init__(root, transform, pre_transform)

        task_inilialized = task.task_class(data_loc=dataset_loc)
        self.task_name = task_inilialized.task_name
        task_inilialized.load_dataset(difficulty)
        graph_problems = task_inilialized.problem_set

        self.data_list = []
        for gp in graph_problems:
            if gp["exact_answer"] == None:
                continue
            if len(gp["graph"]) == 2:
                gp["graph"] = nx.disjoint_union(gp["graph"][0], gp["graph"][1])

            node_dict = {j: i for i, j in enumerate(gp["graph"].nodes())}

            # Convert NetworkX graph to PyG Data object
            data = from_networkx(gp["graph"])

            # Add node features
            data.x = torch.ones(data.num_nodes, 1)

            # Add label
            data.y = torch.tensor([gp["exact_answer"]], dtype=torch.long)
            
            # Store original NetworkX graph for GraphToken mode
            data.nx_graph = gp["graph"]
            
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 634dadd (Add experiment scripts and results logging for GraphToken tasks)
            data.question = "Provide your answer as a single integer value.\n"
            if self.task_name == Tasks.CONNECTED.name:
                data.question += "Identify the connected components in the given graph.\n"
            elif self.task_name == Tasks.DIAMETER.name:
                data.question += "What is the diameter of the given graph?\n"
            elif self.task_name == Tasks.DISTANCE.name:
                data.question = "What is the distance between the specified nodes in the given graph?\n"
            elif self.task_name == Tasks.GED.name:
                data.question += "What is the graph edit distance between the two graphs provided?\n"
            elif self.task_name == Tasks.MCP.name:
                data.question += "What is the size of the maximum clique in the given graph?\n"
            elif self.task_name == Tasks.MCS.name:
                data.question += "What is the size of the maximum common subgraph between the two graphs provided?\n"
            elif self.task_name == Tasks.MIS.name:
                data.question += "What is the size of the maximum independent set in the given graph?\n"
            elif self.task_name == Tasks.MVC.name:
                data.question += "What is the size of the minimum vertex cover in the given graph?\n"
            elif self.task_name == Tasks.NEIGHBOR.name:
                data.question += "How many neighbors does the specified node have in the given graph?\n"
            elif self.task_name == Tasks.TSP.name:
                data.question += "What is the length of the shortest possible route that visits each node exactly once and returns to the origin node in the given graph?\n"
            else:
                data.question += f"What is the solution for the {self.task_name} problem on the given graph?\n"
            data.question += "Answer:"
<<<<<<< HEAD
=======
            # Store question text if available
            if "question" in gp:
                data.question = gp["question"]
>>>>>>> fe4049b (Add wip version)
=======
>>>>>>> 634dadd (Add experiment scripts and results logging for GraphToken tasks)

            # Handle source/target nodes
            if "node1" in gp:
                gp["source"] = gp["node1"]
            if "node2" in gp:
                gp["target"] = gp["node2"]

            if "source" in gp:
                gp["source"] = node_dict[gp["source"]]
                source_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                source_mask[gp["source"]] = True
                data.source = source_mask

            if "target" in gp:
                gp["target"] = node_dict[gp["target"]]
                target_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                target_mask[gp["target"]] = True
                data.target = target_mask

            self.data_list.append(data)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

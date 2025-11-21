from typing import Type
from enum import Enum, StrEnum

import torch
import networkx as nx
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import from_networkx

import tasks

class Difficulty(StrEnum):
    EASY = 'easy'
    HARD = 'hard'

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
    def value(self) -> Type[tasks.NPTask]:
        # for better type hinting
        return self._value_().task_name


class GPDataset(Dataset):
    def __init__(self, task: Tasks = Tasks.CONNECTED, difficulty: Difficulty=Difficulty.EASY, root=None, transform=None, pre_transform=None):
        self.task_name = task.value.task_name
        self.difficulty = difficulty
        super().__init__(root, transform, pre_transform)
        
        dataset_loc =  'dataset'
        task_inilialized = task.value(data_loc=dataset_loc)
        task_inilialized.load_dataset(difficulty)
        graph_problems = task_inilialized.problem_set
        
        self.data_list = []
        for gp in graph_problems:
            if gp['exact_answer'] == None:
                continue
            if len(gp['graph']) == 2:
                gp['graph'] = nx.disjoint_union(gp['graph'][0], gp['graph'][1])
            
            node_dict = {j:i for i, j in enumerate(gp['graph'].nodes())}
            
            # Convert NetworkX graph to PyG Data object
            data = from_networkx(gp['graph'])
            
            # Add node features
            data.x = torch.ones(data.num_nodes, 1)
            
            # Add label
            data.y = torch.tensor([gp['exact_answer']], dtype=torch.long)
            
            # Handle source/target nodes
            if 'node1' in gp:
                gp['source'] = gp['node1']
            if 'node2' in gp:
                gp['target'] = gp['node2']
            
            if 'source' in gp:
                gp['source'] = node_dict[gp['source']]
                source_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                source_mask[gp['source']] = True
                data.source = source_mask
            
            if 'target' in gp:
                gp['target'] = node_dict[gp['target']]
                target_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                target_mask[gp['target']] = True
                data.target = target_mask
            
            self.data_list.append(data)
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]
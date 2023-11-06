import numpy as np
import matplotlib.pyplot as plt

from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.maze import LatticeMaze, TargetedLatticeMaze, SolvedMaze
from maze_dataset.plotting import MazePlot

from torch.utils import data

# implementing my own maze dataset class,

from torch.utils.data import Dataset
from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.tokenization import MazeTokenizer, TokenizationMode



def find_from_right(lst, item):
  try:
      index = lst[::-1].index(item)  # Reverse the list and find the item
      return len(lst) - 1 - index  # Adjust the index for the original list
  except ValueError:
      return None  # Item not found


class CustomMazeDataset(Dataset):
    def __init__(self, grid_n=5, include_maze=False):
        # Initialization code goes here
        self.include_maze = include_maze
        self.grid_n = grid_n
        self.tokenizer = MazeTokenizer(max_grid_size=grid_n)
        self.vocab_size = self.tokenizer.vocab_size

        self.token_arr = self.tokenizer.token_arr

        self.pad_token = self.tokenizer.padding_token_index

        self.vocab_map = self.tokenizer.tokenizer_map.get
        pass

    def __getitem__(self, index):
        # Replace this with your code to generate a sample given an index
        # right now everything is random. I don't know how to create the same maze twice

        start_pos = np.random.randint(self.grid_n,size=2)

        maze = LatticeMazeGenerators.gen_dfs_percolation(grid_shape=np.array([self.grid_n, self.grid_n]),p=0.1)

        goal_pos = np.random.choice([0,self.grid_n-1],2) # randomly choose goal


        tgt_maze: TargetedLatticeMaze = TargetedLatticeMaze.from_lattice_maze(maze, start_pos, goal_pos)
        solved_maze: SolvedMaze = SolvedMaze.from_targeted_lattice_maze(tgt_maze)

        tokens = solved_maze.as_tokens(self.tokenizer)


        ints = list(map(self.vocab_map, tokens))

        goal_index = find_from_right(ints, 2)

        ints_no_goal = ints[:goal_index] + ints[goal_index +3:]

        start_index = goal_index
        if self.include_maze:
          return {'data': ints_no_goal, 'start_index': start_index, 'maze': solved_maze}
        else:
          return {'data': ints_no_goal, 'start_index': start_index}

    def __len__(self):
        # Replace this with your code to return the length of the dataset
        return 1000000000


def collate_fn(batch):
    # Get the maximum sequence length in the batch
    max_len = max(len(item['data']) for item in batch)

    # Pad the sequences and store them in a list
    padded_data = np.array([np.pad(item['data'], (0, max_len - len(item['data'])), 'constant') for item in batch])

    # Store the end indices in a list
    end_indices = np.array([len(item['data']) - 1 for item in batch])

    # Initialize the collated batch with the padded data and end indices
    collated_batch = {'data': padded_data, 'end_index': end_indices}

    # Add any other keys from the batch elements
    for key in batch[0]:
        if key not in collated_batch:
            collated_batch[key] = np.array([item[key] for item in batch])

    return collated_batch





#def numpy_collate(batch):
#  # define collate here, with padding
#
#  return tree_map(np.asarray, data.default_collate(batch))

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

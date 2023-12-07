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
    def __init__(self, grid_n=5, include_maze=False, no_loops=False):
        # Initialization code goes here
        self.include_maze = include_maze
        self.grid_n = grid_n
        self.tokenizer = MazeTokenizer(max_grid_size=grid_n)
        self.vocab_size = self.tokenizer.vocab_size

        self.token_arr = self.tokenizer.token_arr

        self.pad_token = self.tokenizer.padding_token_index

        self.vocab_map = self.tokenizer.tokenizer_map.get
        self.no_loops = no_loops

    def __getitem__(self, index):

        # right now everything is random. I don't know how to create the same maze twice

        start_pos = np.random.randint(self.grid_n, size=2)

        if self.no_loops:
            maze = LatticeMazeGenerators.gen_dfs_percolation(grid_shape=np.array([self.grid_n, self.grid_n]), p=0.)
        else:
            maze = LatticeMazeGenerators.gen_dfs_percolation(grid_shape=np.array([self.grid_n, self.grid_n]), p=0.1)

        # randomly choose goal
        goal_pos = np.random.choice([0, self.grid_n-1], 2)

        tgt_maze: TargetedLatticeMaze = TargetedLatticeMaze.from_lattice_maze(maze, start_pos, goal_pos)
        solved_maze: SolvedMaze = SolvedMaze.from_targeted_lattice_maze(tgt_maze)

        tokens = solved_maze.as_tokens(self.tokenizer)

        ints = list(map(self.vocab_map, tokens))

        goal_index = find_from_right(ints, 2)

        ints_no_goal = ints[:goal_index] + ints[goal_index +3:]

        start_index = goal_index

        if self.no_loops:

            true_probs = []
            true_preds = []
            fork_indx = []

            all_paths = []

            for other_goal in [[0, 0], [0, self.grid_n-1], [self.grid_n-1, 0], [self.grid_n-1, self.grid_n-1]]:
                other_maze: TargetedLatticeMaze = TargetedLatticeMaze.from_lattice_maze(maze, start_pos, other_goal)
                other_solved_maze: SolvedMaze = SolvedMaze.from_targeted_lattice_maze(other_maze)

                other_tokens = other_solved_maze.as_tokens(self.tokenizer)
                other_ints = list(map(self.vocab_map, other_tokens))
                other_goal_index = find_from_right(ints, 2)
                assert other_goal_index == goal_index
                other_path_ints = other_ints[other_goal_index + 4:]
                all_paths.append(other_path_ints)

            true_path = ints_no_goal[start_index+1:-1]

            n_valid_paths = 4
            valid_paths = [0,1,2,3]

            for step in range(len(true_path)):
                # path prediction stuff
                true_pred = np.zeros(self.vocab_size)

                for j in valid_paths:
                    next_step = all_paths[j][step]
                    true_pred[next_step] += 1/len(valid_paths)
                true_preds.append(true_pred)

                # probe related stuff

                valid_paths = []
                for i, other_path in enumerate(all_paths):
                    if step < len(other_path) and other_path[step] == true_path[step]:
                        if step > 0:
                            if other_path[step-1] == true_path[step-1]:
                                valid_paths.append(i)
                            else:
                                assert other_path[step] == true_path[step] == 7
                        else:
                            valid_paths.append(i)

                if step == 0:
                    assert len(valid_paths) == n_valid_paths

                true_prob = np.array([1/len(valid_paths) if i in valid_paths else 0. for i in range(len(all_paths))])
                true_probs.append(true_prob)

                if len(valid_paths) != n_valid_paths:
                    try:
                        assert len(valid_paths) < n_valid_paths
                    except:
                        print(all_paths)
                        print(valid_paths)
                        print(true_path)
                        print(true_probs)
                        raise Exception(f"len(valid_paths) = {len(valid_paths)}, n_valid_paths = {n_valid_paths}")
                    assert sum(true_probs[-2] > 0) - sum(true_probs[-1] > 0.) == n_valid_paths - len(valid_paths)
                    fork_indx.append(step-1)
                    n_valid_paths = len(valid_paths)

        out = {'data': ints_no_goal, 'start_index': start_index}

        if self.include_maze:
            out['maze'] = solved_maze

        if self.no_loops:
            out['true_probs'] = true_probs
            out['fork_indx'] = fork_indx
            out['true_preds'] = true_preds

        return out

    def __len__(self):
        # arbitrary since we have infinite data
        return 1000000000


def collate_fn(batch):
    keys_to_pad = ['data', 'true_probs', 'fork_indx', 'true_preds']
    collated_batch = {}
    # Iterate through each key in the batch that needs to be padded
    for key in keys_to_pad & batch[0].keys():
        # Get the maximum sequence length in the batch
        max_len = max(len(item[key]) for item in batch)

        # Pad the sequences and store them in a list
        padded_data = np.array([np.pad(item[key], [(0, max_len - len(item[key]))] + [(0, 0)] * (len(np.array(item[key]).shape)-1), 'constant', constant_values=-1) for item in batch])

        # Store the end indices in a list
        end_indices = np.array([len(item[key]) - 1 for item in batch])

        # add the padded data and end indices to the collated batch
        collated_batch[key] = padded_data
        collated_batch[key + '_end_index'] = end_indices

    # for backwards compatibility, add extra end index of data
    collated_batch['end_index'] = collated_batch['data_end_index']

    # Add any other keys from the batch elements, which haven't been padded
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

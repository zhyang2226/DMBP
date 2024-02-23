import torch
import pprint
import numpy as np
from typing import *

class Batch:

    def __init__(self, *args, **kwargs):
        self.__dict__.update(dict(*args, **kwargs))

    def __setattr__(self, key: str, value: Any) -> None:
        """Set self.key = value."""
        self.__dict__[key] = value

    def __getattr__(self, key: str) -> Any:
        """Return self.key."""
        return getattr(self.__dict__, key)

    def __contains__(self, key: str) -> bool:
        """Return key in self."""
        return key in self.__dict__

    def __getstate__(self) -> Dict[str, Any]:
        """Pickling interface.

        Only the actual data are serialized for both efficiency and simplicity.
        """
        state = {}
        for k, v in self.items():
            if isinstance(v, Batch):
                v = v.__getstate__()
            state[k] = v
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Unpickling interface.

        At this point, self is an empty Batch instance that has not been
        initialized, so it can safely be initialized by the pickle state.
        """
        self.__init__(**state)  # type: ignore

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def __getitem__(self, index: Union[str, Union[slice, int, np.ndarray, List[int]]]) -> 'Batch':
        """Return self[index]."""
        if isinstance(index, str):
            return self.__dict__[index]
        batch = Batch()
        for k, v in self.items():
            batch[k] = v[index]
        return batch

    def __setitem__(self, index: Union[str, Union[slice, int, np.ndarray, List[int]]], value: Any) -> None:
        """Assign value to self[index]."""
        if isinstance(index, str):
            self.__dict__[index] = value
        else:
            assert isinstance(value, Batch)
            for k, v in value.items():
                self[k][index] = v

    def __repr__(self) -> str:
        """Return str(self)."""
        s = self.__class__.__name__ + "(\n"
        flag = False
        for k, v in self.items():
            rpl = "\n" + " " * (6 + len(k))
            obj = pprint.pformat(v).replace("\n", rpl)
            s += f"    {k}: {obj},\n"
            flag = True
        if flag:
            s += ")"
        else:
            s = self.__class__.__name__ + "()"
        return s

    def to_numpy(self) -> 'Batch':
        """Change all torch.Tensor to numpy.ndarray in-place."""
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self[k] = v.detach().cpu().numpy()
        return self

    def to_torch(self, dtype: torch.dtype = torch.float32, device: str = "cpu") -> 'Batch':
        """Change all numpy.ndarray to torch.Tensor in-place."""
        for k, v in self.items():
            self[k] = torch.as_tensor(v, dtype=dtype, device=device)
        return self

    @staticmethod
    def cat(batches: List["Batch"], axis: int = 0) -> "Batch":
        """Concatenate a list of Batch object into a single new batch."""
        if isinstance(list(batches[0].values())[0], np.ndarray):
            cat_func = np.concatenate
        else:
            cat_func = torch.cat
        batch = Batch()
        for k in batches[0].keys():
            batch[k] = cat_func([b[k] for b in batches], axis=axis)
        return batch

    @staticmethod
    def stack(batches: List["Batch"], axis: int = 0) -> "Batch":
        """Stack a list of Batch object into a single new batch."""
        if isinstance(list(batches[0].values())[0], np.ndarray):
            stack_func = np.stack
        else:
            stack_func = torch.stack
        batch = Batch()
        for k in batches[0].keys():
            batch[k] = stack_func([b[k] for b in batches], axis=axis)
        return batch

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def shape(self) -> List[int]:
        data_shape = []
        for v in self.__dict__.values():
            try:
                data_shape.append(list(v.shape))
            except AttributeError:
                data_shape.append([])
        return list(map(min, zip(*data_shape))) if len(data_shape) > 1 \
            else data_shape[0]

    def split(self, size: int, shuffle: bool = True, merge_last: bool = False) -> Iterator["Batch"]:
        length = len(self)
        assert 1 <= size  # size can be greater than length, return whole batch
        if shuffle:
            indices = np.random.permutation(length)
        else:
            indices = np.arange(length)
        merge_last = merge_last and length % size > 0
        for idx in range(0, length, size):
            if merge_last and idx + size + size >= length:
                yield self[indices[idx:]]
                break
            yield self[indices[idx:idx + size]]


class SampleBatch(Batch):
    def sample(self, batch_size):
        length = len(self)
        assert 1 <= batch_size

        indices = np.random.randint(0, length, batch_size)
        return self[indices]
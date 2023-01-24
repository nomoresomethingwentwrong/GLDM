from enum import Enum, auto
from pickle import loads, dumps
import torch

class Colors(Enum):
    GREEN = auto()
    BLUE = auto()

torch.save(Colors.BLUE, 'enum_test.pt')
a = Colors.BLUE is loads(dumps(Colors.BLUE))
b = Colors["BLUE"] == loads(dumps(Colors.BLUE))
c = loads(dumps(Colors.BLUE)) in [Colors.BLUE, Colors.GREEN]
d = torch.load('enum_test.pt') in [Colors.BLUE, Colors.GREEN] 
print(a, b, c, d)

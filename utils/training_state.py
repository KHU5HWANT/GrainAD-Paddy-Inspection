"""Atomic, trusted local training-state checkpoints and cooperative epoch pauses."""
import os, random
from pathlib import Path
import numpy as np
import torch


def save_training(path, **state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state['rng']={'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],'python':random.getstate(),'numpy':np.random.get_state()}
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(state,temporary);os.replace(temporary,path)


def load_training(path,device):
    # Only load locally created, trusted checkpoints, which contain RNG metadata.
    state=torch.load(path,map_location=device,weights_only=False)
    rng=state.get('rng')
    if rng:
        torch.set_rng_state(rng['torch'].cpu());random.setstate(rng['python']);np.random.set_state(rng['numpy'])
        if torch.cuda.is_available() and rng['cuda']:
            torch.cuda.set_rng_state_all([x.cpu() for x in rng['cuda']])
    return state


def pause_requested():
    flag=os.environ.get('GRAINAD_PAUSE_FILE')
    return bool(flag and Path(flag).exists())

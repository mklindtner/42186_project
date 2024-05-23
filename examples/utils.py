import pyro
from tqdm import tqdm

def run_inference(model, guide, game_info, obs, num_steps=2000, optimizer=None, lr=0.01, verbose=True):
    
    pyro.clear_param_store()

    if optimizer is None:
        optimizer = pyro.optim.Adam({"lr": lr})

    svi = pyro.infer.SVI(model, guide, optimizer, loss=pyro.infer.Trace_ELBO())

    pbar = tqdm(range(num_steps))
    for step in pbar:
        loss = svi.step(game_info, obs)
        if step % 50 == 0 and verbose:
            pbar.set_description("Loss = %f" % loss)
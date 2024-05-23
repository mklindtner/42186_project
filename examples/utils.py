import pyro

def run_inference(model, game_info, obs, num_steps=5000, guide=None, optimizer=None, lr=0.01, verbose=True):
    
    pyro.clear_param_store()

    if optimizer is None:
        optimizer = pyro.optim.Adam({"lr": lr})

    if guide is None:
        guide = pyro.contrib.autoguide.AutoNormal(model)

    svi = pyro.infer.SVI(model, guide, optimizer, loss=pyro.infer.Trace_ELBO())

    for step in range(num_steps):
        loss = svi.step(game_info, obs)
        if step % 100 == 0 and verbose:
            print(f"Step {step} : loss = {loss}")
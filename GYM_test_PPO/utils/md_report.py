from parameters import *

def create_md_summary(gym_id: str, name: str, folder: str, seed: float)-> None:
    """
    Function that create a MarkDown report for parameters used during training
    """
    report = folder + name + ".md"
    
    with open(report, 'w') as file:
        file.write("# Enviroment: " + gym_id + "\n")
        
        if int(seed) == SEED:
            file.write(f"Seed: {seed}, (deterministic)\n")
        else:
            file.write(f"Seed: {seed}, (random)\n")


        file.write("## Training parameters\n")

        file.write(f"- Total epoch: {MAX_EPOCH}\n")
        file.write(f"- Number of enviroments: {n_env}\n")
        file.write(f"- Timestep for collecting data T = {n_step}\n")
        file.write(f"- Total data for each loop: {BATCH_SIZE}\n")
        file.write(f"- Update epoch K = {K_EPOCHS}\n")
        file.write(f"- Minibatch size {MINI_BATCH_SIZE}\n\n")

        file.write("## Hyperparameters\n")
        file.write(f"* Discount factor: {GAMMA}\n")
        file.write(f"* GAE lambda: {GAE_LAMBDA}\n")
        file.write(f"* Learning rate: {LR}\n")
        file.write(f"* Clipping factor: {CLIP}\n")
        file.write(f"* Loss: c1 = {VALUE_COEFF}; c2 = {ENTROPY_COEF}\n")

if __name__ == "__main__":
    create_md_summary("Summary test", " Testissimo", "")
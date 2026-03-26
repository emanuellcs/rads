import os
import time
import torch
import arc_agi
import multiprocessing as mp
from typing import List, Dict, Any

# Internal RADS Modules
from orchestrator.shared_memory import IPCMemoryManager, IPCWorkerClient
from orchestrator.gpu_batch_server import start_gpu_server_process, POISON_PILL
from models.trm_verifier import get_compiled_trm
from agent.epistemic_foraging import EpistemicForager
from agent.physics_simulator import ARCPhysicsSimulator, compile_dummy_hypothesis, ARCGameState
from agent.mcts import MCTSAgent

# Configuration Constants
NUM_WORKERS = 4
BATCH_SIZE = 64
MCTS_SIMULATIONS = 2000

def cpu_swarm_worker(interface: Dict[str, Any], games_to_play: List[str]):
    """
    The main logic loop for a single CPU worker in the Swarm.
    Handles physical game interaction, epistemic probing, and MCTS planning.
    """
    worker_id = interface["worker_id"]
    print(f"[Worker {worker_id}] Initializing Swarm node for {len(games_to_play)} games.")
    
    # 1. Initialize the IPC client to talk to the GPU server
    ipc_client = IPCWorkerClient(interface)
    
    # 2. Initialize the official ARC-AGI-3 environment toolkit
    arc = arc_agi.Arcade()
    
    for game_name in games_to_play:
        print(f"[Worker {worker_id}] Starting game: {game_name}")
        
        # Load the environment in high-speed headless mode (+2K FPS)
        env = arc.make(game_name)
        
        # Initialize the Epistemic Forager for this specific game
        forager = EpistemicForager(max_resets=3, beam_size=16)
        
        # --- PHASE 1: EPISTEMIC FORAGING ---
        # Execute the 4-step Minimum Viable Probe (MVP) to ground spatial coordinates
        forager.execute_mvp_sequence(env)
        
        # NOTE: In the full pipeline, the 8B Masked Diffusion Prior would now ingest the 
        # MVP trajectory and generate B=16 Python rule hypotheses. For this execution script, 
        # we will route to a compiled dummy hypothesis to demonstrate the MCTS integration.
        active_hypothesis = compile_dummy_hypothesis
        
        # --- PHASE 2: INTERNAL WORLD MODEL COMPILATION ---
        simulator = ARCPhysicsSimulator(rule_hypothesis_fn=active_hypothesis)
        mcts = MCTSAgent(ipc_client=ipc_client, physics_simulator=simulator, c_puct=1.25)
        
        # --- PHASE 3: PRAGMATIC EXECUTION ---
        while not env.is_game_over() and not env.is_win():
            # Convert the raw JSON environment frame into our lightweight ARCGameState
            raw_frame = env.get_current_frame()
            current_state = ARCGameState(
                grid=raw_frame["grid"], 
                agent_r=raw_frame["agent_y"], 
                agent_c=raw_frame["agent_x"]
            )
            
            # Run the Decoupled Thinking Loop (Zero physical action cost)
            # This will query the GPU server thousands of times via the IPC client
            best_action = mcts.search(root_state=current_state, num_simulations=MCTS_SIMULATIONS)
            
            # Execute the verified action in the real environment
            env.step(best_action)
            
        print(f"[Worker {worker_id}] Finished game: {game_name}. Win status: {env.is_win()}")

def main():
    """
    Top-level orchestrator. Sets up the hardware, spawns the IPC queues, 
    and manages the lifecycle of the GPU server and CPU workers.
    """
    print("=== RADS ARC-AGI-3 Execution Engine ===")
    
    # Ensure offline compliance
    os.environ["WANDB_MODE"] = "offline"
    
    # 1. Allocate Shared Memory IPC Architecture
    print("Allocating zero-copy IPC memory segments...")
    ipc_manager = IPCMemoryManager(num_slots=256, state_bytes=16384, num_workers=NUM_WORKERS)
    
    # 2. Load and Compile the 7M TRM Verifier onto the GPU
    print("Compiling Tiny Recursive Verifier (CUDA Graphs)...")
    compiled_trm = get_compiled_trm(device="cuda:0")
    
    # 3. Spawn the dedicated GPU Batch Server
    gpu_interface = ipc_manager.get_gpu_server_interface()
    gpu_process = start_gpu_server_process(gpu_interface, compiled_trm)
    
    # 4. Discover ARC-AGI-3 Games
    arc = arc_agi.Arcade()
    all_games = arc.get_available_games() # In competition, this fetches the 110 private games
    
    # Partition games evenly across the CPU Swarm workers
    chunk_size = len(all_games) // NUM_WORKERS
    game_chunks = [all_games[i:i + chunk_size] for i in range(0, len(all_games), chunk_size)]
    
    # Ensure any remaining games are added to the last chunk
    if len(game_chunks) > NUM_WORKERS:
        game_chunks[-2].extend(game_chunks[-1])
        game_chunks.pop()
        
    worker_interfaces = ipc_manager.get_worker_interfaces()
    
    # 5. Launch the CPU Swarm
    print(f"Launching {NUM_WORKERS} MCTS CPU Workers...")
    start_time = time.time()
    
    worker_processes = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=cpu_swarm_worker, args=(worker_interfaces[i], game_chunks[i]))
        p.start()
        worker_processes.append(p)
        
    # 6. Wait for all games to be completed
    for p in worker_processes:
        p.join()
        
    elapsed_time = time.time() - start_time
    print(f"All Swarm workers finished in {elapsed_time:.2f} seconds.")
    
    # 7. Graceful Shutdown
    print("Sending poison pill to GPU Batch Server...")
    gpu_interface["request_queue"].put(POISON_PILL)
    gpu_process.join(timeout=5.0)
    
    # 8. Output Final Scorecard
    print("=== Final Competition Scorecard ===")
    scorecard = arc.get_scorecard()
    print(scorecard)

if __name__ == "__main__":
    main()
import numpy as np
import pickle
from datetime import datetime

from Objects.MeshGrid import MeshGrid

class Simulation():
    def __init__(self, width:int=10, height:int=10, cell_size:float=1, dt:float=0.01):
        self.grid = MeshGrid(width=width, height=height, cell_size=cell_size)
        self.grid.generateMesh()
        self.dt = dt  # Delta Time
        self.time = 0.0 # Length of simulation in seconds
        self.frames = []
        

    def saveFrame(self):
        """
        Save current state as a frame with timestamp.
        """
        self.grid.updateVerticesFromPoints()
        
        velocities = np.array([point.velocity for point in self.grid.points])
        
        frame = {
            "timestamp": self.time,
            "vertices": self.grid.vertices.copy(),
            "velocities": velocities.copy()
        }

        self.frames.append(frame)
    

    def step(self):
        """
        Perform one simulation step.
        """
        # Update positions based on velocity
        for point in self.grid.points:
            point.position += point.velocity * self.dt
        
        # Update velocitites based on physics simulation
        for point in self.grid.points:
            # NEEDS TO BE IMPLEMENTED TO REAL SIMULATION WITH ODE SOLVERS. TEMP PLACEHODLDER!!!!!!
            x, y, z = point.position

            wave_height = 0.1 * np.sin(x * 0.5 + self.time * 2) * np.cos(y * 0.5 + self.time * 2)
            target_z = wave_height

            # Spring Damper effect to move point to target height
            point.velocity[2] += (target_z - z) * 0.1
            point.velocity[2] *= 0.95
        
        self.time += self.dt
        
        # Update vertices for next step
        self.grid.updateVerticesFromPoints()
    

    def run(self, duration:float=10.0, save_interval:float=0.1, output_file:str="simulation_frames.pkl"):
        """
        Run simulation and save frames.
        
        Args:
            duration: Total simulation time in seconds.
            save_interval: Time between saved frames in seconds.
            output_file: Path to output file.
        """
        num_steps = int(duration / self.dt)
        save_every = int(save_interval / self.dt)
        
        print(f"Running simulation for {duration} seconds...")
        print(f"Saving every {save_interval} seconds ({save_every} steps)")
        
        # Save initial frame
        self.saveFrame()
        
        for step in range(num_steps):
            self.step()
            
            # Save frame at intervals
            if step % save_every == 0:
                self.saveFrame()
                if step % (save_every * 10) == 0:
                    print(f"Time: {self.time:.2f}s, Frame: {len(self.frames)}")
        
        # Save final frame
        if len(self.frames) == 0 or self.frames[-1]['timestamp'] != self.time:
            self.saveFrame()
        
        # Save to file
        self.saveToFile(output_file)
        print(f"\nSimulation complete. Saved {len(self.frames)} frames to {output_file}")
    

    def saveToFile(self, filename:str="simulation_frames.pkl"):
        """
        Save all frames to a pickle file.
        """

        with open(filename, "wb") as f:
            pickle.dump({
                "frames": self.frames,
                "metadata": {
                    "width": self.grid.width,
                    "height": self.grid.height,
                    "cell_size": self.grid.cell_size,
                    "dt": self.dt,
                    "num_frames": len(self.frames),
                    "duration": self.time,
                    "created": datetime.now().isoformat()
                }
            }, f)
    

    @staticmethod
    def loadFromFile(filename:str="simulation_frames.pkl"):
        """
        Load simulation frames from file.
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)

        return data


if __name__ == "__main__":
    sim = Simulation(width=20, height=20, cell_size=1, dt=0.01)
    sim.run(duration=20.0, save_interval=0.1, output_file="simulation_frames.pkl")


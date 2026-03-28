# 🏎️ f1RaceSim: Volumetric Aero & ROS3 Control 🏁

https://youtu.be/xBuxTSqUMP4

-----
**Real-time CFD-lite Visualization & High-Fidelity F1 Physics in MuJoCo**

## 📺 Watch the Interactive Simulator in action!

<p align="center">
  <a href="https://www.youtube.com/watch?v=xBuxTSqUMP4">
    <img src="https://img.youtube.com/vi/xBuxTSqUMP4/maxresdefault.jpg" alt="f1racesim Demo Video" width="800" style="border-radius:10px;">
    <br>
    <b>▶️ Click to Watch the Simulation: F1 Volumetric Flow Demo</b>
  </a>
</p>

---
## 🏎️ Description

**f1racesim** is a technical showcase of real-time aerodynamics and low-latency robotics control. By bridging **MuJoCo's** contact dynamics with a custom **Volumetric Vortex Lattice Method (VLM)**, this project visualizes the invisible: the complex wake and downforce generation of a modern Formula 1 car.

The control layer is powered by **ROS3**, utilizing the `mirage` messaging protocol to achieve near-zero input lag between an **Xbox Controller 🎮** and the simulated vehicle.

## 🌪️ The Volumetric Aero Engine

Unlike static aero models, **f1racesim** calculates flow interaction dynamically:

  * **Wake Turbulence:** 3,000+ particles react to the car's geometry in real-time.
  * **Ground Effect Simulation:** Downforce scales with `q_s` (dynamic pressure) and proximity to the track.
  * **Visual Velocity Mapping:** Particle colors shift from **Red (Compression/High Speed)** to **Blue (Wake/Low Speed)** based on local velocity vectors.

## 🕹️ Xbox Integration (ROS3)

The simulation subscribes to `/xbox/controller` using a high-frequency ROS3 node.

  * **Left Stick:** Precision steering.
  * **Triggers:** Normalized 0-1 throttle/brake mapping.
  * **Right Stick:** Live "Wind Tunnel" adjustments (change wind speed/direction on the fly).

## 🛠️ Build & Run

### Dependencies

  * `mujoco` (Physics)
  * `glfw3` (Window/Input)
  * `ros3`, `mirage`, `quicksand` (Middleware)

### Compilation

```bash
gcc main.c -lmujoco -lglfw -lros3 -lmirage -lquicksand -lm -o f1_sim
```

### Execution

```bash
./f1_sim models/f1_car.xml
```

## 📜 Acknowledgments

  * **Physics:** MuJoCo / Todorov / Washington / Google DeepMind
  * **Inspiration:** Our work on the University Rover Challenge robotics competition


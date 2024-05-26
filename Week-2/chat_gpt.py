import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
g = 9.81  # gravity (m/s^2)
e = 0.8   # coefficient of restitution (elasticity)
h0 = 10   # initial height (meters)
v0 = 0    # initial velocity (m/s)
dt = 0.05 # time step (seconds)

# Lists to store time, height, and velocity
times = [0]
heights = [h0]
velocities = [v0]

# Calculate the motion
while len(heights) < 1000:  # simulate for a limited number of steps
    t = times[-1] + dt
    v = velocities[-1] - g * dt
    h = heights[-1] + v * dt

    if h <= 0:
        h = 0
        v = -v * e  # reverse and reduce the velocity due to the bounce

    times.append(t)
    heights.append(h)
    velocities.append(v)

# Create the animation
fig, ax = plt.subplots()
ax.set_xlim(0, max(times))
ax.set_ylim(0, h0 + 2)
line, = ax.plot([], [], 'o', markersize=10)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data(times[frame], heights[frame])
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(times), init_func=init, blit=True)
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.title('Bouncing Ball Animation')
plt.show()
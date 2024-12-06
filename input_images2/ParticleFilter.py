import cv2
import numpy as np

def load_frames(base_path, num_frames):
    for i in range(num_frames):
        # filename = f"pres_debate/{i:03d}.jpg" #1
        filename = f"pres_debate_noisy/{i:03d}.jpg"
        frame = cv2.imread(filename)
        if frame is not None:
            yield frame
        else:
            print(f"Failed to load frame: {filename}")

def calculate_histogram(frame, position, hist_size=13, mask_radius=12):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(position[0]), int(position[1])), mask_radius, 255, -1)
    hist = cv2.calcHist([frame], [0, 1, 2], mask, [hist_size]*3, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist

def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

num_particles = 200  
# initial_pos = np.array([575, 441]) #1
initial_pos = np.array([708, 457])
particles = np.tile(initial_pos, (num_particles, 1)) + np.random.randn(num_particles, 2) * 10  
velocity = np.zeros((num_particles, 2))  
target_hist = None  

def particle_filter(frames, particles, velocity):
    global target_hist
    last_known_position = initial_pos
    for frame in frames:
        if target_hist is None: 
            target_hist = calculate_histogram(frame, initial_pos)

        particles += velocity + np.random.randn(num_particles, 2) * 10
        velocity += np.random.randn(num_particles, 2) * 2

        weights = np.zeros(num_particles)
        for i, particle in enumerate(particles):
            particle_hist = calculate_histogram(frame, particle)
            weights[i] = 1 - compare_histograms(target_hist, particle_hist) 

        weights += 1e-10  
        weights /= weights.sum()

        N_eff = 1.0 / np.sum(np.square(weights))
        if N_eff < num_particles / 2:
            indices = np.random.choice(range(num_particles), size=num_particles, replace=True, p=weights)
            particles = particles[indices]
            velocity = velocity[indices] 
        else:
            if np.max(weights) < 0.1:
                particles = np.tile(last_known_position, (num_particles, 1)) + np.random.randn(num_particles, 2) * 20
                velocity = np.zeros((num_particles, 2))

        for particle in particles:
            cv2.circle(frame, (int(particle[0]), int(particle[1])), 1, (0, 255, 0), -1)

        last_known_position = particles[np.argmax(weights)]  

        cv2.imshow('Particle Filter Tracking', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# frames = load_frames('/pres_debate', 166) #1
frames = load_frames('/pres_debate_noisy', 100)
particle_filter(frames, particles, velocity)


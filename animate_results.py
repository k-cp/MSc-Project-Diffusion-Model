import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


#  name of the folder

EXPERIMENT_FOLDER = "kmflow_re1000_rs256_ddim_conditional_new"

# Select among
# kmflow_re1000_rs256_ddim_conditional_new
# kmflow_re1000_rs256_ddim_recons_conditional_log
# kmflow_re1000_rs256_ddim_log
# kmflow_re1000_rs256_ddim_new



BASE_EXP_NAME = "guided_recons_u3232_t240_r30"

# Guidance weight (e.g., 0.0, 0.5, 1.0)
W = 0.0

# Output format (select to'mp4' or 'gif')
OUTPUT_FORMAT = 'mp4'

# Layout (select 'combined' or 'individual')
LAYOUT = 'combined'



parent_dir = f"experiments/{EXPERIMENT_FOLDER}/{BASE_EXP_NAME}_w{W}"
num_batches = 64

if not os.path.exists(parent_dir):
    print(f"Error: The folder '{parent_dir}' was not found in the root directory.")
    print("Check your BASE_EXP_NAME and W values.")
    exit()

all_inputs, all_samples, all_references = [], [], []

print(f"Loading data from: {parent_dir}")


for i in range(num_batches):
    batch_folder = os.path.join(parent_dir, f"sample_batch{i}")
    input_path = os.path.join(batch_folder, "input_arr.npy")
    sample_path = os.path.join(batch_folder, "sample_arr_run_0_it0.npy")
    reference_path = os.path.join(batch_folder, "reference_arr.npy")
    
    if os.path.exists(input_path):
        all_inputs.append(np.load(input_path))
        all_samples.append(np.load(sample_path))
        all_references.append(np.load(reference_path))

inputs = np.concatenate(all_inputs, axis=0)
samples = np.concatenate(all_samples, axis=0)
references = np.concatenate(all_references, axis=0)
total_frames = samples.shape[0]
print(f"Loaded {total_frames} frames.")

def get_display_frame(arr, frame_idx):
    frame = arr[frame_idx]
    return frame[0] if frame.ndim == 3 else frame

def save_animation(ani, filename_base):
    filename = f"{filename_base}.{OUTPUT_FORMAT}"
    writer = 'ffmpeg' if OUTPUT_FORMAT == 'mp4' else 'pillow'
    try:
        ani.save(filename, writer=writer, fps=20)
        print(f"Success! Saved: {filename}")
    except Exception as e:
        print(f"Error saving {filename}: {e}")


if LAYOUT == 'combined':
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ims = [axes[i].imshow(get_display_frame(data, 0), cmap='jet') 
           for i, data in enumerate([inputs, samples, references])]
    
    for ax, title in zip(axes, [" Input", "Diffusion Recons", "Reference"]):
        ax.set_title(title)
        ax.axis('off')

    def update(frame):
        ims[0].set_data(get_display_frame(inputs, frame))
        ims[1].set_data(get_display_frame(samples, frame))
        ims[2].set_data(get_display_frame(references, frame))
        return ims

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=50, blit=True)
    save_animation(ani, f"{BASE_EXP_NAME}_w{W}_combined")
    plt.close()

elif LAYOUT == 'individual':
    def render_single(data, title, suffix):
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(get_display_frame(data, 0), cmap='jet')
        ax.set_title(title)
        ax.axis('off')
        ani = animation.FuncAnimation(fig, lambda f: [im.set_data(get_display_frame(data, f))], 
                                      frames=total_frames, interval=50, blit=False)
        save_animation(ani, f"{BASE_EXP_NAME}_w{W}_{suffix}")
        plt.close()

    render_single(inputs, " Input", "input")
    render_single(samples, "Diffusion Reconstruction", "sample")
    render_single(references, "Ground Truth", "ref")
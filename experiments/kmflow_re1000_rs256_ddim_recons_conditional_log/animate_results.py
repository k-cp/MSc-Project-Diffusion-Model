import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Choose output format: 'mp4' or 'gif'
OUTPUT_FORMAT = 'mp4'

# Choose  layout: 'combined' (side-by-side) or 'individual' (3 separate files)
LAYOUT = 'combined'



parent_dir = "./guided_recons_u3232_t240_r30_w0.0"
num_batches = 64

all_inputs, all_samples, all_references = [], [], []

print("Loading and stitching batches...")

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
print(f"Full sequence assembled! Total frames: {total_frames}")

def get_display_frame(arr, frame_idx):
    frame = arr[frame_idx]
    if frame.ndim == 3: 
        return frame[0]
    return frame

def save_animation(ani, filename_base):
    try:
        if OUTPUT_FORMAT == 'mp4':
            filename = f"{filename_base}.mp4"
            ani.save(filename, writer='ffmpeg', fps=20)
        else:
            filename = f"{filename_base}.gif"
            ani.save(filename, writer='pillow', fps=20)
        print(f"Success! Saved: {filename}")
    except Exception as e:
        print(f"Error saving {filename_base}: {e}")
        if OUTPUT_FORMAT == 'mp4':
            print("Hint: MP4 requires 'ffmpeg' installed on your system. Switch OUTPUT_FORMAT to 'gif' if it fails.")


if LAYOUT == 'combined':
    print(f"Rendering combined {OUTPUT_FORMAT.upper()} ")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Fluid Velocity Field Reconstruction ($\mathbf{u}$)", fontsize=16)

    im_input = axes[0].imshow(get_display_frame(inputs, 0), cmap='jet')
    axes[0].set_title("Sparse Input")
    axes[0].axis('off')

    im_sample = axes[1].imshow(get_display_frame(samples, 0), cmap='jet')
    axes[1].set_title("Diffusion Reconstruction")
    axes[1].axis('off')

    im_ref = axes[2].imshow(get_display_frame(references, 0), cmap='jet')
    axes[2].set_title("Ground Truth")
    axes[2].axis('off')

    plt.tight_layout()

    def update_combined(frame_idx):
        im_input.set_data(get_display_frame(inputs, frame_idx))
        im_sample.set_data(get_display_frame(samples, frame_idx))
        im_ref.set_data(get_display_frame(references, frame_idx))
        return [im_input, im_sample, im_ref]

    ani = animation.FuncAnimation(fig, update_combined, frames=total_frames, interval=50, blit=True)
    save_animation(ani, "fluid_reconstruction_combined")
    plt.close()

elif LAYOUT == 'individual':
    print(f"Rendering 3 individual {OUTPUT_FORMAT.upper()} files (this might take a minute)...")
    
    def render_single(data, title, filename_base):
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.suptitle(title, fontsize=16)
        im = ax.imshow(get_display_frame(data, 0), cmap='jet')
        ax.axis('off')
        
        def update_single(frame_idx):
            im.set_data(get_display_frame(data, frame_idx))
            return [im]
            
        ani = animation.FuncAnimation(fig, update_single, frames=total_frames, interval=50, blit=True)
        save_animation(ani, filename_base)
        plt.close()

    render_single(inputs, "Sparse Input", "fluid_01_input")
    render_single(samples, "Diffusion Reconstruction", "fluid_02_sample")
    render_single(references, "Ground Truth Reference", "fluid_03_reference")

else:
    print("Error: LAYOUT must be 'combined' or 'individual'")
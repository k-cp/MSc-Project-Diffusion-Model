import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os
import glob

def compute_ke_spectrum(vorticity_field):
    """Computes the 1D kinetic energy spectrum E(k) from a 2D vorticity field."""
    N = vorticity_field.shape[0]
    
    # 2D Fast Fourier Transform
    w_hat = np.fft.fftshift(np.fft.fft2(vorticity_field))
    
    kx = np.fft.fftshift(np.fft.fftfreq(N)) * N
    ky = np.fft.fftshift(np.fft.fftfreq(N)) * N
    KX, KY = np.meshgrid(kx, ky)
    
    K_sq = KX**2 + KY**2
    K_sq[K_sq == 0] = 1e-10
    
    E_2d = 0.5 * (np.abs(w_hat)**2) / K_sq
    E_2d[N//2, N//2] = 0 
    
    K_mag = np.sqrt(K_sq)
    K_int = np.round(K_mag).astype(int)
    
    max_k = N // 2
    E_k = np.zeros(max_k)
    for i in range(1, max_k):
        E_k[i] = np.sum(E_2d[K_int == i])
        
    return np.arange(1, max_k), E_k[1:]

def collect_and_average_data(batch_dir_pattern, file_name):
    """
    Finds all matching .npy files across your batch folders and collects them.
    """
    search_path = os.path.join(batch_dir_pattern, file_name)
    file_list = glob.glob(search_path)
    
    if not file_list:
        raise FileNotFoundError(f"Could not find any files matching: {search_path}")
        
    print(f"Found {len(file_list)} batch folders containing '{file_name}'.")
    
    fields = []
    for f in file_list:
        data = np.load(f)
        # If sequence data is saved, grab the final frame
        if data.ndim == 3:
            data = data[-1]  
        fields.append(data)
        
    return fields

def plot_fluid_statistics(experiment_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.rcParams.update({'font.size': 12, 'axes.linewidth': 1.5})
    
    batch_pattern = os.path.join(experiment_dir, "sample_batch*")
    
    methods = {
        # FIX: Change 'sample_arr_run_0_it2.npy' to 'sample_arr_run_0_it0.npy'
        'Conditional Method': {'file': 'sample_arr_run_0_it0.npy', 'color': 'red',        'linestyle': '-'},
        'Reference':          {'file': 'reference_arr.npy',        'color': 'mediumblue', 'linestyle': '-'}
    }
    
    for name, info in methods.items():
        try:
            fields = collect_and_average_data(batch_pattern, info['file'])
        except FileNotFoundError as e:
            print(e)
            continue
            
        # ==========================================
        # Plot (a) Kinetic Energy Spectrum E(k)
        # ==========================================
        total_E_k = None
        k = None
        for field in fields:
            k, E_k = compute_ke_spectrum(field)
            if total_E_k is None:
                total_E_k = E_k
            else:
                total_E_k += E_k
        avg_E_k = total_E_k / len(fields)
        ax1.loglog(k, avg_E_k, color=info['color'], linestyle=info['linestyle'], linewidth=1.5, label=name)
        
        # ==========================================
        # Plot (b) Vorticity Distribution p(w)
        # ==========================================
        # Concatenate all pixels from all batches for a highly accurate distribution
        all_vorticity = np.concatenate([f.flatten() for f in fields])
        kde = gaussian_kde(all_vorticity, bw_method=0.1)
        
        # Define x-axis range for vorticity 
        w_range = np.linspace(-10, 10, 500)
        p_w = kde(w_range)
        ax2.plot(w_range, p_w, color=info['color'], linestyle=info['linestyle'], linewidth=2, label=name)

    # --- Formatting Ax1: Kinetic Energy Spectrum ---
    ax1.set_xlabel(r'$k$', fontsize=14)
    ax1.set_ylabel(r'$E(k)$', fontsize=14)
    ax1.set_xlim(left=1)
    ax1.grid(True, which="major", linestyle='-.', color='gray', alpha=0.5)
    ax1.set_title('(a) Kinetic energy spectrum', y=-0.2)
    ax1.legend(loc='lower left', framealpha=0.9)
    
    # --- Formatting Ax2: Vorticity Distribution ---
    ax2.set_xlabel(r'$\boldsymbol{\omega}$', fontsize=14)
    ax2.set_ylabel(r'$p(\boldsymbol{\omega})$', fontsize=14)
    ax2.set_xlim(-10, 10)
    ax2.set_ylim(0, 0.12)
    ax2.grid(True, which="major", linestyle='-.', color='gray', alpha=0.5)
    ax2.set_title('(b) Vorticity distribution', y=-0.2)
    ax2.legend(loc='upper right', framealpha=0.9)

    # Save and display
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2) 
    save_target = os.path.join(experiment_dir, "conditional_vs_reference_stats.png")
    plt.savefig(save_target, dpi=300, bbox_inches='tight')
    print(f"\nPlot successfully saved to: {save_target}")
    plt.show()


if __name__ == "__main__":
    
    # Change this line: remove 'repos/Diffusion-based-Fluid-Super-resolution/'
    exp_path = "experiments/kmflow_re1000_rs256_ddim_conditional_new/guided_recons_u3232_t240_r30_w0.0"
    
    plot_fluid_statistics(exp_path)
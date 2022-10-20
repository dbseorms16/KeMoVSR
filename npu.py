import numpy as np
import os
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

def plot_kernel(out_k_np, savepath, gt_k_np=None):
    plt.clf()
    if gt_k_np is None:
        ax = plt.subplot(111)
        im = ax.imshow(out_k_np, vmin=out_k_np.min(), vmax=out_k_np.max())
        plt.colorbar(im, ax=ax)
    else:

        ax = plt.subplot(121)
        im = ax.imshow(gt_k_np, vmin=gt_k_np.min(), vmax=gt_k_np.max())
        plt.colorbar(im, ax=ax)
        ax.set_title('GT Kernel')

        ax = plt.subplot(122)
        im = ax.imshow(out_k_np, vmin=gt_k_np.min(), vmax=gt_k_np.max())
        plt.colorbar(im, ax=ax)
        # ax.set_title('Kernel PSNR: {:.2f}'.format(calculate_kernel_psnr(out_k_np, gt_k_np)))

    # plt.show()
    plt.savefig(savepath)


def build_kernel(sig1, sig2, theta):
    kernel_size = 21
    kernel_radius = kernel_size // 2
    kernel_range = np.linspace(-kernel_radius, kernel_radius, kernel_size)

    # horizontal_range = kernel_range[None].repeat((self.kernel_size, 1))
    # vertical_range = kernel_range[:, None].repeat((1, self.kernel_size))
    horizontal_range, vertical_range = np.meshgrid(kernel_range, kernel_range)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    cos_theta_2 = cos_theta ** 2
    sin_theta_2 = sin_theta ** 2

    sigma_x_2 = 2.0 * (sig1 ** 2)
    sigma_y_2 = 2.0 * (sig2 ** 2)

    a = cos_theta_2 / sigma_x_2 + sin_theta_2 / sigma_y_2
    b = sin_theta * cos_theta * (1.0 / sigma_y_2 - 1.0 / sigma_x_2)
    c = sin_theta_2 / sigma_x_2 + cos_theta_2 / sigma_y_2

    gaussian = lambda x, y: np.exp((- (a * (x ** 2) + 2.0 * b * x * y + c * (y ** 2))))
    kernel = gaussian(horizontal_range, vertical_range)
    kernel = kernel / kernel.sum()
    return kernel

kernel_folder = '../pretrained_models/Mixed/Vid4.npy'
# kernel_folder = '../pretrained_models/Mixed/REDS.npy'
kernel_preset = np.load(kernel_folder)
print(kernel_preset.shape)

# for a in range(kernel_preset.shape[0]):
#     save_ker_path = os.path.join('./vimeo_kernels', '0kernel_{:s}.png'.format(str(a)))
#     plot_kernel(kernel_preset[a], save_ker_path)

results=[]
sig1s = np.arange(0.4, 2, 0.1)
sig2s = np.arange(0.4, 2, 0.1)
thetas = np.arange(0.1, 0.8, 0.05)
# thetas = [0, 0.7854]

array1 = []
array2 = []
mini = 1000000
for a in range(kernel_preset.shape[0]):
    GT = kernel_preset[a]
    
    mini = 1000000
    
    for sig1 in sig1s:
        for sig2 in sig2s:
            for theta in thetas:
                sig1 = round(sig1, 2)
                sig2 = round(sig2, 2)
                theta = round(theta, 2)
                testkernel = build_kernel(sig1, sig2, theta)
                
                results = abs(GT - testkernel).sum()
                if results < mini:
                    ssig1 = sig1
                    ssig2 = sig2
                    ttheta= theta
                    mini = results
                    
    testkernel = build_kernel(ssig1, ssig2, ttheta)
    print('frame:',a,'sig1',ssig1,'sig2',ssig2,'theta',ttheta,'   =',mini)
    
    array1.append(ssig1)
    array2.append(ssig2)
    
    save_ker_path = os.path.join('./estkernels', '0kernel_{:s}.png'.format(str(a)))
    plot_kernel(testkernel, save_ker_path)

    save_ker_path = os.path.join('./new_vimeo_kernels', '0kernel_{:s}.png'.format(str(a)))
    plot_kernel(GT, save_ker_path)
    
np.save('Vid4sig1.npz',array1)
np.save('Vid4sig2.npz',array2)
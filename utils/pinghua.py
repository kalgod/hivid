import numpy as np

def modify_score(llm_array):
    sigma=5
    kernel_size=5
    k=kernel_size//2
    x=np.arange(-k,k+1)
    gaussian_kernel=np.exp(-0.5*(x/sigma)**2)
    gaussian_kernel/=gaussian_kernel.sum()
    llm_array = np.array([float(x) for x in llm_array])
    smoothed_array = np.convolve(llm_array, gaussian_kernel, mode='same')
    return smoothed_array.tolist()


def main():
    arr = [85, 90, 95, 95, 90, 80, 75, 70, 65, 60]
    marr = modify_score(arr)
    print(marr)
    return 0

main()
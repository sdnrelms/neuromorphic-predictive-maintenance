import scipy.io
import matplotlib.pyplot as plt

files = ["97.mat", "105.mat", "118.mat", "130.mat"]

for f in files:
    path = f"datasets/cwru/raw/{f}"
    mat = scipy.io.loadmat(path)
    print(f"\n--- {f} ---")
    for key in mat.keys():
        if not key.startswith("__"):
            print(f"  Değişken: {key}, boyut: {mat[key].shape}")


# Her sınıftan bir örnek sinyali görselleştirelim (ilk 1000 örnek yeterli, tüm sinyali çizmeye gerek yok)
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

mat97 = scipy.io.loadmat("datasets/cwru/raw/97.mat")
mat105 = scipy.io.loadmat("datasets/cwru/raw/105.mat")
mat118 = scipy.io.loadmat("datasets/cwru/raw/118.mat")
mat130 = scipy.io.loadmat("datasets/cwru/raw/130.mat")

axs[0].plot(mat97["X097_DE_time"][:1000], color='green')
axs[0].set_title("Normal")

axs[1].plot(mat105["X105_DE_time"][:1000], color='orange')
axs[1].set_title("Inner Race Arızası")

axs[2].plot(mat118["X118_DE_time"][:1000], color='red')
axs[2].set_title("Ball Arızası")

axs[3].plot(mat130["X130_DE_time"][:1000], color='purple')
axs[3].set_title("Outer Race Arızası")

plt.tight_layout()
plt.savefig("datasets/cwru/cwru_signals_comparison.png")
plt.show()
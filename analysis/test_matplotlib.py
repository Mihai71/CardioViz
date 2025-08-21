import matplotlib
import matplotlib.pyplot as plt

print("Matplotlib version:", matplotlib.__version__)
fig, ax = plt.subplots()
ax.plot([0,1,2], [0,1,0])
ax.set_title("OK")
fig.savefig("static/plots/_smoke_test.png")
print("Saved to static/plots/_smoke_test.png")